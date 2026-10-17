import sys

import numpy as np

import sim
from learn import Learner, LearnerConfig


numGames = 500  # More games use more RAM


# Makes one game, gets called once for every game
def makeEnv(idx):
    env = sim.Env(sim.GameMode.soccar, playersPerTeam=1)
    env.rewards = [
        # Main
        (sim.AirReward(), 0.085),
        (sim.FaceBallReward(), 0.05),
        (sim.VelocityPlayerToBallReward(), 1),
        (sim.TouchBallReward(), 5),
        #(sim.VelocityBallToGoalReward(), 5 ),
        #(sim.GoalReward(), 150),
    ]

    env.terminals = [sim.NoTouchCondition(15), sim.GoalScoreCondition()]
    env.obsBuilder = sim.AdvancedObs()
    env.actionParser = sim.LookupTableAction()
    env.mutator = sim.KickoffMutator()
    return env


# Our metrics, the arrays are every player (or every game) at once
def stepCallback(envSet, report):
    stats = envSet.getStats()
    game = stats["game"]

    report.avg("Player/In Air Ratio", ~stats["onGround"])
    report.avg("Player/Ball Touch Ratio", stats["touched"])
    report.avg("Player/Demoed Ratio", stats["demoed"])

    vel = stats["vel"]
    report.avg("Player/Speed", np.linalg.norm(vel, axis=1))
    toBall = stats["ballPos"][game] - stats["pos"]
    dirToBall = toBall / np.linalg.norm(toBall, axis=1, keepdims=True)
    report.avg("Player/Speed Towards Ball", np.maximum(0, (vel * dirToBall).sum(axis=1)))

    report.avg("Player/Boost", stats["boost"])

    touched = stats["touched"]
    report.avg("Player/Touch Height", stats["ballPos"][game[touched], 2])

    scored = stats["goalScored"]
    report.avg("Game/Goal Speed", np.linalg.norm(stats["ballVel"][scored], axis=1))


def main():
    cfg = LearnerConfig()
    cfg.numGames = numGames
    cfg.tickSkip = 8
    cfg.actionDelay = cfg.tickSkip - 1
    cfg.randomSeed = -1

    cfg.ppo.timestepsPerIteration = 50_000
    cfg.ppo.batchSize = 50_000
    cfg.ppo.miniBatchSize = cfg.ppo.timestepsPerIteration // 2  # Lower this if too much VRAM is used
    cfg.ppo.epochs = 1

    # This is the scale for normalized entropy, it doesn't need changing if you add more actions
    cfg.ppo.entropyScale = 0.035
    cfg.ppo.gaeGamma = 0.99
    cfg.ppo.policyLR = 2e-4
    cfg.ppo.criticLR = 2e-4

    cfg.ppo.sharedHead.layerSizes = [512, 512, 512]
    cfg.ppo.policy.layerSizes = [512, 512, 512]
    cfg.ppo.critic.layerSizes = [512, 512, 512]

    cfg.sendMetrics = True  # wandb
    cfg.renderMode = "render" in sys.argv
    cfg.ppo.deterministic = cfg.renderMode

    learner = Learner(makeEnv, cfg)
    learner.start(stepCallback)


if __name__ == "__main__":
    main()
