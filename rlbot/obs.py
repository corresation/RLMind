import numpy as np

import common_values
from game_state import GameState, Player
from phys_state import PhysState

PLAYER_OBS_SIZE = 29

# Same as RLGC::AdvancedObs
class AdvancedObs:
    POS_COEF = 1 / 5000
    VEL_COEF = 1 / 2300
    ANG_VEL_COEF = 1 / 3

    @staticmethod
    def get_obs_size(players_per_team: int) -> int:
        return 9 + common_values.NUM_ACTIONS + common_values.BOOST_LOCATIONS_AMOUNT + PLAYER_OBS_SIZE * players_per_team * 2

    def add_player_to_obs(self, obs: list, player: Player, inv: bool, ball: PhysState):
        phys = player.phys.invert() if inv else player.phys
        rot_t = phys.rot_mat.T # Same as RotMat::Dot()

        obs += [
            phys.pos * self.POS_COEF,
            phys.forward,
            phys.up,
            phys.vel * self.VEL_COEF,
            phys.ang_vel * self.ANG_VEL_COEF,
            rot_t @ phys.ang_vel * self.ANG_VEL_COEF,

            rot_t @ (ball.pos - phys.pos) * self.POS_COEF,
            rot_t @ (ball.vel - phys.vel) * self.VEL_COEF,

            [player.boost / 100, player.is_on_ground, player.has_flip_or_jump, player.is_demoed, player.has_jumped],
        ]

    def build_obs(self, player: Player, state: GameState, prev_action: np.ndarray) -> np.ndarray:
        inv = player.team == common_values.ORANGE_TEAM

        ball = state.ball.invert() if inv else state.ball
        pads = state.get_boost_pads(inv)
        pad_timers = state.get_boost_pad_timers(inv)

        obs = [
            ball.pos * self.POS_COEF,
            ball.vel * self.VEL_COEF,
            ball.ang_vel * self.ANG_VEL_COEF,
            prev_action,
            np.where(pads > 0, 1, 1 / (1 + pad_timers)),
        ]

        self.add_player_to_obs(obs, player, inv, ball)

        teammates, opponents = [], []
        for other in state.players:
            if other.car_id == player.car_id:
                continue
            self.add_player_to_obs(teammates if other.team == player.team else opponents, other, inv, ball)

        return self.combine(obs, teammates, opponents)

    def combine(self, obs: list, teammates: list, opponents: list) -> np.ndarray:
        return np.concatenate([np.asarray(x, dtype=np.float32) for x in obs + teammates + opponents])

# Same as RLGC::AdvancedObsPadded, zeros for missing players up to 2 teammates and 3 opponents
class AdvancedObsPadded(AdvancedObs):
    MAX_TEAMMATES = 2
    MAX_OPPONENTS = 3
    OBS_SIZE = AdvancedObs.get_obs_size(3)

    def combine(self, obs: list, teammates: list, opponents: list) -> np.ndarray:
        num_teammates = sum(np.size(x) for x in teammates) // PLAYER_OBS_SIZE
        num_opponents = sum(np.size(x) for x in opponents) // PLAYER_OBS_SIZE
        teammates = teammates + [np.zeros(PLAYER_OBS_SIZE * max(self.MAX_TEAMMATES - num_teammates, 0))]
        opponents = opponents + [np.zeros(PLAYER_OBS_SIZE * max(self.MAX_OPPONENTS - num_opponents, 0))]
        return super().combine(obs, teammates, opponents)
