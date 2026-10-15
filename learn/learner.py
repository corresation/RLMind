import glob
import json
import math
import msvcrt
import os
import random
import shutil
import socket
import time
from dataclasses import dataclass, field

import numpy as np
import torch

import sim
from .metrics import Report, Wandb
from .ppo import PPO, PPOConfig
from .versions import PolicyVersions, SkillTrackerConfig


projectFolder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
renderAddress = ("127.0.0.1", 9273)  # RocketSimVis's default port


@dataclass
class LearnerConfig:
    numGames: int = 300

    tickSkip: int = 8
    actionDelay: int = 7

    # Plays the newest save in RocketSimVis instead of training, switching to newer saves as they appear
    renderMode: bool = False
    renderTimeScale: float = 1  # 2 = twice real speed

    ppo: PPOConfig = field(default_factory=PPOConfig)

    # Checkpoints are saved here as timestep-numbered subfolders, empty disables saving
    checkpointFolder: str = "checkpoints"
    tsPerSave: int = 1_000_000
    checkpointsToKeep: int = 8

    randomSeed: int = -1  # -1 uses the current time
    device: str = "auto"  # auto, cuda or cpu

    standardizeReturns: bool = True
    maxReturnSamples: int = 150

    addRewardsToMetrics: bool = True
    maxRewardSamples: int = 50  # Games sampled per step for reward metrics
    rewardSampleRandInterval: int = 8  # Reward metrics are sampled on about 1 in this many steps

    # Saved copies of the policy, needed for the skill tracker
    savePolicyVersions: bool = False
    tsPerVersion: int = 25_000_000
    maxOldVersions: int = 32

    # Elo ratings from games against old versions (turns on savePolicyVersions)
    skillTracker: SkillTrackerConfig = field(default_factory=SkillTrackerConfig)

    sendMetrics: bool = True  # wandb
    metricsProjectName: str = "rlmind"
    metricsGroupName: str = "unnamed-runs"
    metricsRunName: str = "rlmind-run"

    collisionMeshesPath: str = os.path.join(projectFolder, "collision_meshes")


class WelfordStat:
    def __init__(self):
        self.mean = 0.0
        self.var = 0.0
        self.count = 0

    def increment(self, samples):
        for sample in samples:
            delta = float(sample) - self.mean
            deltaN = delta / (self.count + 1)
            self.mean += deltaN
            self.var += delta * deltaN * self.count
            self.count += 1

    def std(self):
        if self.count < 2:
            return 1.0
        var = self.var / (self.count - 1)
        return math.sqrt(var) if var != 0 else 1.0


displayRows = [
    "Average Step Reward",
    "Policy Entropy",
    "Mean KL Divergence",
    "SB3 Clip Fraction",
    "Explained Variance",
    "",
    "Policy Update Magnitude",
    "Critic Update Magnitude",
    "Shared Head Update Magnitude",
    "",
    "Collection Steps/Second",
    "Consumption Steps/Second",
    "Overall Steps/Second",
    "",
    "Collection Time",
    "-Inference Time",
    "-Env Step Time",
    "Consumption Time",
    "-Critic Infer Time",
    "-GAE Time",
    "-PPO Learn Time",
    "",
    "Collected Timesteps",
    "Experience Timesteps",
    "Total Timesteps",
    "Total Iterations",
]


class Learner:
    def __init__(self, makeEnv, cfg, stepCallback=None):
        self.cfg = cfg
        self.stepCallback = stepCallback

        if cfg.renderMode:
            cfg.numGames = 1
            cfg.sendMetrics = cfg.addRewardsToMetrics = False
            cfg.savePolicyVersions = cfg.skillTracker.enabled = False

        sim.init(cfg.collisionMeshesPath)

        print("Learner::Learner():")
        if cfg.randomSeed == -1:
            cfg.randomSeed = int(time.time() * 1000) % 2**31
        torch.manual_seed(cfg.randomSeed)
        self.rng = np.random.default_rng(cfg.randomSeed)
        print(f'\tCheckpoint Save/Load Dir: "{cfg.checkpointFolder}"')

        if cfg.device == "auto":
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(cfg.device)
        print("\tUsing CUDA GPU device..." if self.device.type == "cuda" else "\tUsing CPU device...")

        print("\tCreating envs...")
        self.envSet = sim.EnvSet([makeEnv(i) for i in range(cfg.numGames)], cfg.tickSkip, cfg.actionDelay, cfg.addRewardsToMetrics)

        print("\tMaking PPO learner...")
        self.ppo = PPO(self.envSet.obsSize, self.envSet.numActions, cfg.ppo, self.device)
        assert cfg.ppo.batchSize <= cfg.ppo.timestepsPerIteration, "batchSize can't be bigger than timestepsPerIteration"

        self.returnStat = WelfordStat() if cfg.standardizeReturns else None
        self.totalTimesteps = 0
        self.totalIterations = 0
        self.runId = ""

        if cfg.skillTracker.enabled:
            cfg.savePolicyVersions = True
        self.versions = None
        if cfg.savePolicyVersions:
            assert cfg.checkpointFolder, "Policy versions need a checkpoint folder"
            self.versions = PolicyVersions(self, makeEnv)

        if cfg.checkpointFolder:
            self.load()
            if self.versions:
                self.versions.load(self.totalTimesteps)

        self.metrics = None
        if cfg.sendMetrics:
            print("Initializing MetricSender...")
            self.metrics = Wandb(cfg.metricsProjectName, cfg.metricsGroupName, cfg.metricsRunName, self.runId)
            print(f' > {"Continuing" if self.runId else "Starting"} run with ID : "{self.metrics.runId}"...')
            self.runId = self.metrics.runId
            print(" > MetricSender initalized.")
        print("=" * 40)

        # Steps of episodes that haven't finished yet carry over to the next iteration
        numPlayers = self.envSet.numPlayers
        starts = self.envSet.arenaPlayerStartIdx
        self.playerGame = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, numPlayers)))
        self.steps = {key: [] for key in ("obs", "masks", "actions", "logProbs", "rewards", "terminals")}
        self.stepBase = 0  # Absolute index of steps[...][0]
        self.epStart = np.zeros(numPlayers, np.int64)  # Where each player's current episode starts in steps
        self.epLen = np.zeros(numPlayers, np.int64)
        self.truncNextObs = {}  # (absolute step, player): obs after the truncated step
        self.maxEpisodeLength = int(cfg.ppo.maxEpisodeDuration * (120 / cfg.tickSkip))
        self.renderSave = None
        self.actor = self.ppo.makeActor(numPlayers, self.envSet.obsSize, self.envSet.numActions)

    def collect(self, report):
        cfg, envSet, steps = self.cfg, self.envSet, self.steps
        inferTime = envTime = 0
        completed = numSteps = 0

        while completed < cfg.ppo.timestepsPerIteration:
            t = time.perf_counter()
            envSet.reset()
            envTime += time.perf_counter() - t

            obs = envSet.obs.copy()
            masks = envSet.actionMasks.copy()
            if not np.isfinite(obs).all():
                raise RuntimeError("Obs builder produced a NaN/inf value")

            envSet.stepFirstHalf()

            t = time.perf_counter()
            actions, logProbs = self.actor(obs, masks)
            inferTime += time.perf_counter() - t

            t = time.perf_counter()
            envSet.stepSecondHalf(actions)
            envTime += time.perf_counter() - t

            if self.stepCallback and random.random() <= 0.25:
                self.stepCallback(envSet, report)
            if cfg.addRewardsToMetrics and self.rng.integers(cfg.rewardSampleRandInterval) == 0:
                self.addRewardMetrics(report)

            terminals = envSet.terminals[self.playerGame].astype(np.int8)
            self.epLen += 1
            terminals[(terminals == 0) & (self.epLen >= self.maxEpisodeLength)] = 2  # Too long, truncate (no reset)

            absStep = self.stepBase + len(steps["obs"])
            for p in np.flatnonzero(terminals == 2):
                self.truncNextObs[(absStep, p)] = envSet.obs[p].copy()

            ended = terminals != 0
            completed += self.epLen[ended].sum()
            self.epLen[ended] = 0

            steps["obs"].append(obs)
            steps["masks"].append(masks)
            steps["actions"].append(actions)
            steps["logProbs"].append(logProbs)
            steps["rewards"].append(envSet.rewards.copy())
            steps["terminals"].append(terminals)
            numSteps += 1

        report["Inference Time"] = inferTime
        report["Env Step Time"] = envTime
        return numSteps * envSet.numPlayers

    def addRewardMetrics(self, report):
        lastRewards = self.envSet.getLastRewards()
        games = self.rng.integers(len(lastRewards), size=min(len(lastRewards), self.cfg.maxRewardSamples))
        for name, val in zip(self.envSet.rewardNames, lastRewards[games].mean(0)):
            report.addAvg("Rewards/" + name, val)

    def takeEpisodes(self):
        steps = self.steps
        terminals = np.stack(steps["terminals"])
        numSteps, numPlayers = terminals.shape

        hasTerm = terminals != 0
        lastTerm = numSteps - 1 - np.argmax(hasTerm[::-1], axis=0)
        lastTerm[~hasTerm.any(0)] = -1
        counts = np.where(lastTerm >= self.epStart, lastTerm - self.epStart + 1, 0)

        playerIdx = np.repeat(np.arange(numPlayers), counts)
        offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        stepIdx = np.repeat(self.epStart, counts) + offsets

        # Gather step by step, stacking every stored step gets slow once long episodes carry over
        order = np.argsort(stepIdx, kind="stable")
        bounds = np.searchsorted(stepIdx[order], np.arange(numSteps + 1))
        exp = {}
        for key, vals in steps.items():
            out = np.empty((len(stepIdx),) + vals[0].shape[1:], vals[0].dtype)
            for t in np.flatnonzero(np.diff(bounds)):
                rows = order[bounds[t]:bounds[t + 1]]
                out[rows] = vals[t][playerIdx[rows]]
            exp[key] = out
        truncRows = np.flatnonzero(exp["terminals"] == 2)
        exp["truncNextObs"] = np.array(
            [self.truncNextObs.pop((self.stepBase + stepIdx[r], playerIdx[r])) for r in truncRows], np.float32
        ).reshape(len(truncRows), self.envSet.obsSize)
        exp["truncRows"] = truncRows

        newStart = np.where(counts > 0, lastTerm + 1, self.epStart)
        cut = newStart.min()
        for vals in steps.values():
            del vals[:cut]
        self.epStart = newStart - cut
        self.stepBase += cut
        return exp

    def iteration(self):
        cfg = self.cfg
        report = Report()

        collectStart = time.perf_counter()
        stepsCollected = self.collect(report)
        exp = self.takeEpisodes()
        collectionTime = time.perf_counter() - collectStart

        consumeStart = time.perf_counter()
        dev = self.device
        tObs = torch.from_numpy(exp["obs"]).to(dev)
        values = self.ppo.valuesBatched(tObs).cpu().numpy()
        truncValues = np.zeros_like(values)
        truncValues[exp["truncRows"]] = self.ppo.valuesBatched(torch.from_numpy(exp["truncNextObs"]).to(dev)).cpu().numpy()

        t = time.perf_counter()
        returnStd = self.returnStat.std() if self.returnStat else 1
        advantages, returns, clipPortion = sim.gae(
            exp["rewards"], exp["terminals"], values, truncValues,
            cfg.ppo.gaeGamma, cfg.ppo.gaeLambda, returnStd, cfg.ppo.rewardClipRange
        )
        report["GAE Time"] = time.perf_counter() - t
        report["Clipped Reward Portion"] = clipPortion

        if self.returnStat:
            report["GAE/Returns STD"] = returnStd
            self.returnStat.increment(returns[self.rng.integers(len(returns), size=min(cfg.maxReturnSamples, len(returns)))])
        report["GAE/Avg Return"] = np.abs(returns).mean()
        report["GAE/Avg Advantage"] = np.abs(advantages).mean()
        targetValues = values + advantages
        report["GAE/Avg Val Target"] = np.abs(targetValues).mean()
        report["Explained Variance"] = 1 - np.var(targetValues - values) / max(np.var(targetValues), 1e-8)
        report["Average Step Reward"] = exp["rewards"].mean()
        report["Episode Length"] = len(exp["terminals"]) / max((exp["terminals"] != 0).sum(), 1)

        t = time.perf_counter()
        self.ppo.learn({
            "obs": tObs,
            "masks": torch.from_numpy(exp["masks"]).to(dev).bool(),
            "actions": torch.from_numpy(exp["actions"]).to(dev).long(),
            "logProbs": torch.from_numpy(exp["logProbs"]).to(dev),
            "advantages": torch.from_numpy(advantages).to(dev),
            "targetValues": torch.from_numpy(targetValues).to(dev),
        }, report)
        report["PPO Learn Time"] = time.perf_counter() - t
        consumptionTime = time.perf_counter() - consumeStart

        report["Collection Time"] = collectionTime
        report["Consumption Time"] = consumptionTime
        report["Collection Steps/Second"] = stepsCollected / collectionTime
        report["Consumption Steps/Second"] = stepsCollected / consumptionTime
        report["Overall Steps/Second"] = stepsCollected / (collectionTime + consumptionTime)

        prevTimesteps = self.totalTimesteps
        self.totalTimesteps += stepsCollected
        self.totalIterations += 1
        report["Collected Timesteps"] = stepsCollected
        report["Total Timesteps"] = self.totalTimesteps
        report["Total Iterations"] = self.totalIterations

        if self.versions:
            self.versions.onIteration(report, self.totalTimesteps, prevTimesteps)
        return report, prevTimesteps

    def start(self, stepCallback=None):
        self.stepCallback = stepCallback or self.stepCallback
        cfg = self.cfg
        print("Learner::Start():")
        print(f"\tObs size: {self.envSet.obsSize}")
        print(f"\tAction amount: {self.envSet.numActions}")
        if cfg.renderMode:
            print("\t(Render mode enabled)")
            return self.render()
        print("Press 'Q' to save and quit!")

        while True:
            report, prevTimesteps = self.iteration()

            quit = False
            while msvcrt.kbhit():
                if msvcrt.getwch().upper() == "Q":
                    quit = True

            if cfg.checkpointFolder and (quit or self.totalTimesteps // cfg.tsPerSave > prevTimesteps // cfg.tsPerSave):
                self.save()

            report.finish()
            if self.metrics:
                self.metrics.send(report)
            report.display(displayRows)

            if quit:
                return

    # Plays the newest save from any build folder in RocketSimVis, start RocketSimVis first
    def render(self):
        cfg = self.cfg
        print("Press 'Q' to quit!")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        stepTime = cfg.tickSkip / 120 / cfg.renderTimeScale
        nextStep = time.perf_counter()
        lastCheck = -math.inf

        while True:
            if time.perf_counter() - lastCheck > 10:
                lastCheck = time.perf_counter()
                saves = glob.glob(os.path.join(projectFolder, "out", "build", "*", "checkpoints", "*", "models.pt"))
                if saves:
                    newest = os.path.dirname(max(saves, key=os.path.getmtime))
                    if newest != self.renderSave:
                        self.renderSave = newest
                        cfg.checkpointFolder = os.path.dirname(newest)
                        self.load()

            self.envSet.reset()
            obs, masks = self.envSet.obs.copy(), self.envSet.actionMasks.copy()
            self.envSet.stepFirstHalf()
            actions, _ = self.actor(obs, masks)
            self.envSet.stepSecondHalf(actions)
            sock.sendto(json.dumps(self.envSet.getRenderState(0)).encode(), renderAddress)

            while msvcrt.kbhit():
                if msvcrt.getwch().upper() == "Q":
                    return

            nextStep += stepTime
            time.sleep(max(0, nextStep - time.perf_counter()))

    def checkpointDirs(self):
        if not os.path.isdir(self.cfg.checkpointFolder):
            return []
        return sorted(int(name) for name in os.listdir(self.cfg.checkpointFolder) if name.isdigit())

    def save(self):
        folder = os.path.join(self.cfg.checkpointFolder, str(self.totalTimesteps))
        print(f'Saving to folder "{folder}"...')
        os.makedirs(folder, exist_ok=True)
        torch.save(self.ppo.state(), os.path.join(folder, "models.pt"))
        stats = {
            "totalTimesteps": self.totalTimesteps, "totalIterations": self.totalIterations, "runId": self.runId,
            "tickSkip": self.cfg.tickSkip, "actionDelay": self.cfg.actionDelay,
        }
        if self.returnStat:
            stats["returnStat"] = vars(self.returnStat)
        if self.versions:
            stats["skillRatings"] = self.versions.curRatings
            self.versions.save()
        with open(os.path.join(folder, "stats.json"), "w") as f:
            json.dump(stats, f, indent=4)

        saved = self.checkpointDirs()
        if self.cfg.checkpointsToKeep != -1:
            for ts in saved[:-self.cfg.checkpointsToKeep]:
                shutil.rmtree(os.path.join(self.cfg.checkpointFolder, str(ts)))
        print(" > Done.")

    def load(self):
        print(f'Loading most recent checkpoint in "{self.cfg.checkpointFolder}"...')
        saved = self.checkpointDirs()
        if not saved:
            print(" > No checkpoints found, starting new model.")
            return
        folder = os.path.join(self.cfg.checkpointFolder, str(saved[-1]))
        print(f' > Loading checkpoint "{folder}"...')
        self.ppo.loadState(torch.load(os.path.join(folder, "models.pt"), map_location=self.device))
        with open(os.path.join(folder, "stats.json")) as f:
            stats = json.load(f)
        self.totalTimesteps = stats["totalTimesteps"]
        self.totalIterations = stats["totalIterations"]
        self.runId = stats["runId"]
        if self.returnStat and "returnStat" in stats:
            vars(self.returnStat).update(stats["returnStat"])
        if self.versions and "skillRatings" in stats:
            self.versions.curRatings = stats["skillRatings"]
        print(" > Done.")
