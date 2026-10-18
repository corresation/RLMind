import json
import os
import shutil
from dataclasses import dataclass

import numpy as np
import torch

import sim
from .ppo import policyProbs, sampleActions


@dataclass
class SkillTrackerConfig:
    enabled: bool = False

    # Games used for rating, don't put this much higher than your CPU thread count
    numArenas: int = 16

    simTime: float = 45  # Seconds simulated per rating run/game
    maxSimTime: float = 240

    updateInterval: int = 16  # Iterations between rating runs
    ratingInc: float = 5  # Rating change scale per goal
    initialRating: float = 0

    # Off by default since learning optimizes the random policy
    deterministic: bool = False


def modeName(teams):
    counts = [(teams == 0).sum(), (teams == 1).sum()]
    return f"{min(counts)}v{max(counts)}"


# Saved copies of the policy every tsPerVersion steps & Elo ratings from games against them
class PolicyVersions:
    def __init__(self, learner, makeEnv):
        self.learner = learner
        self.ppo = learner.ppo
        cfg = learner.cfg
        self.skill = cfg.skillTracker
        self.folder = os.path.join(cfg.checkpointFolder, "policy_versions")
        self.versions = []  # dicts of timesteps, models, ratings
        self.curRatings = {}
        self.iterationsSinceRan = 0
        self.curGoals = 0
        self.continuation = None  # (old version index, new team, sim time) when the last run needs more goals

        if self.skill.enabled:
            def skillMakeEnv(idx):
                env = makeEnv(idx)
                env.rewards = []
                env.terminals = [sim.GoalScoreCondition()]
                env.mutator = sim.FuzzedKickoffMutator()
                return env

            self.envSet = sim.EnvSet([skillMakeEnv(i) for i in range(self.skill.numArenas)], cfg.tickSkip, cfg.actionDelay, False)
            starts = self.envSet.arenaPlayerStartIdx
            self.playerGame = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, self.envSet.numPlayers)))
            self.teams = self.envSet.getStats()["team"]
            self.gameModes = [modeName(self.teams[self.playerGame == g]) for g in range(len(starts))]

    def rating(self, ratings, mode):
        return ratings.setdefault(mode, self.skill.initialRating)

    def add(self, timesteps):
        self.versions.append({"timesteps": timesteps, "models": self.ppo.copyPolicy(), "ratings": dict(self.curRatings)})
        self.versions.sort(key=lambda v: v["timesteps"])
        del self.versions[:-self.learner.cfg.maxOldVersions]

    def onIteration(self, report, totalTimesteps, prevTimesteps):
        cfg = self.learner.cfg
        if totalTimesteps // cfg.tsPerVersion > prevTimesteps // cfg.tsPerVersion or prevTimesteps == 0:
            self.add(totalTimesteps)

        if self.skill.enabled:
            self.iterationsSinceRan += 1
            if self.iterationsSinceRan >= self.skill.updateInterval and self.versions:
                self.iterationsSinceRan = 0
                self.runSkillMatches(report)

    def updateRatings(self, winner, loser, mode):
        winnerRating, loserRating = self.rating(winner, mode), self.rating(loser, mode)
        expected = 1 / (10 ** ((loserRating - winnerRating) / 400) + 1)
        winner[mode] = winnerRating + self.skill.ratingInc * (1 - expected)
        loser[mode] = loserRating + self.skill.ratingInc * (expected - 1)

    @torch.no_grad()
    def runSkillMatches(self, report):
        skill, envSet, rng = self.skill, self.envSet, self.learner.rng
        if self.continuation:
            oldIdx, newTeam, totalSimTime = self.continuation
        else:
            oldIdx, newTeam, totalSimTime = rng.integers(len(self.versions)), int(rng.integers(2)), 0
            envSet.reset()
        self.continuation = None

        old = self.versions[oldIdx]
        isNew = self.teams == newTeam
        dev = self.ppo.device
        newMask = torch.from_numpy(isNew).to(dev)
        temp = self.ppo.cfg.policyTemperature
        print(f"Running skill matches (simTime={skill.simTime})...")
        prevRatings = dict(self.curRatings)

        stepTime = self.learner.cfg.tickSkip / 120
        t = 0
        while t < skill.simTime and totalSimTime < skill.maxSimTime and self.curGoals < envSet.numGames:
            envSet.reset()
            obs = torch.from_numpy(envSet.obs.copy()).to(dev)
            masks = torch.from_numpy(envSet.actionMasks.copy()).to(dev).bool()
            envSet.stepFirstHalf()

            actions = np.zeros(envSet.numPlayers, np.int32)
            for models, players, mask in ((self.ppo.models, isNew, newMask), (old["models"], ~isNew, ~newMask)):
                probs = policyProbs(models, obs[mask], masks[mask], temp)
                actions[players] = sampleActions(probs, skill.deterministic)[0].int().cpu().numpy()
            envSet.stepSecondHalf(actions)

            stats = envSet.getStats()
            for g in np.flatnonzero(stats["goalScored"]):
                ballY = stats["ballPos"][g, 1]
                concededTeam = 0 if ballY < 0 else 1
                mode = self.gameModes[g]
                if concededTeam != newTeam:
                    self.updateRatings(self.curRatings, old["ratings"], mode)
                else:
                    self.updateRatings(old["ratings"], self.curRatings, mode)
                self.curGoals += 1

            t += stepTime
            totalSimTime += stepTime

        for mode, rating in self.curRatings.items():
            delta = rating - prevRatings.get(mode, skill.initialRating)
            print(f" > {mode} = {rating:.2f} ({delta:+.2f})")
            report["Rating/" + mode] = rating

        if self.curGoals < envSet.numGames and totalSimTime < skill.maxSimTime:
            # Not enough goals yet, the same versions keep playing from here next time
            print(f" > Forcing continuation ({self.curGoals}/{envSet.numGames})")
            self.continuation = (oldIdx, newTeam, totalSimTime)
        else:
            self.curGoals = 0

    def save(self):
        os.makedirs(self.folder, exist_ok=True)
        kept = {v["timesteps"] for v in self.versions}
        for name in os.listdir(self.folder):
            if name.isdigit() and int(name) not in kept:
                shutil.rmtree(os.path.join(self.folder, name))

        for version in self.versions:
            folder = os.path.join(self.folder, str(version["timesteps"]))
            os.makedirs(folder, exist_ok=True)
            torch.save({name: model.state_dict() for name, model in version["models"].items()}, os.path.join(folder, "policy.pt"))
            with open(os.path.join(folder, "stats.json"), "w") as f:
                json.dump({"skillRatings": version["ratings"]}, f, indent=4)

    def load(self, curTimesteps):
        if not os.path.isdir(self.folder):
            return
        for name in sorted((n for n in os.listdir(self.folder) if n.isdigit()), key=int):
            timesteps = int(name)
            if timesteps > curTimesteps:
                raise RuntimeError(
                    f"Saved policy version {timesteps} is newer than the current model ({curTimesteps}). "
                    "If you deleted checkpoints, delete that far back in policy_versions too")
            folder = os.path.join(self.folder, name)
            self.add(timesteps)
            version = next(v for v in self.versions if v["timesteps"] == timesteps)
            state = torch.load(os.path.join(folder, "policy.pt"), map_location=self.ppo.device)
            for modelName, model in version["models"].items():
                model.load_state_dict(state[modelName])
            with open(os.path.join(folder, "stats.json")) as f:
                version["ratings"] = json.load(f)["skillRatings"]
        print(f"Loaded {len(self.versions)} policy versions")
