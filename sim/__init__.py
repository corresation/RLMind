from ._sim import *


# What makeEnv() returns for each game
class Env:
    def __init__(self, gameMode=GameMode.soccar, playersPerTeam=1):
        self.gameMode = gameMode
        self.playersPerTeam = playersPerTeam
        self.rewards = []  # (reward, weight) pairs
        self.terminals = []
        self.obsBuilder = None
        self.actionParser = None
        self.mutator = None


# Short names
zerosum = ZeroSumReward
vbtg = VelocityBallToGoalReward
goal = GoalReward


# Playlist split by game index: modes is the % of 1s, 2s and 3s
def getPlayersPerTeam(idx, modes):
    assert sum(modes) == 100, "modes must total to 100"
    r = idx % 100
    if r < modes[0]:
        return 1
    if r < modes[0] + modes[1]:
        return 2
    return 3
