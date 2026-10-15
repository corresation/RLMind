import numpy as np

import common_values
from phys_state import PhysState

# Octane, from RocketSim's CarConfig
HITBOX_HALF_SIZE = np.array([120.507, 86.6994, 38.6591]) / 2
HITBOX_OFFSET = np.array([13.8757, 0, 20.755])

DOUBLEJUMP_MAX_DELAY = 1.25

class Player:
    def __init__(self):
        self.index = -1
        self.car_id = -1
        self.team = -1
        self.phys = PhysState()
        self.boost = 0.0
        self.is_on_ground = False
        self.has_jumped = False
        self.has_flip_or_jump = False
        self.is_demoed = False
        self.is_turtled = False

class GameState:
    def __init__(self, field_info):
        num_pads = field_info.num_boosts
        assert num_pads == common_values.BOOST_LOCATIONS_AMOUNT

        self.ball = PhysState()
        self.players: list[Player] = []

        self.boost_pads = np.ones(num_pads)
        self.boost_pad_timers = np.zeros(num_pads)
        self.pad_cooldowns = np.array([
            common_values.BIG_PAD_COOLDOWN if field_info.boost_pads[i].is_full_boost else common_values.SMALL_PAD_COOLDOWN
            for i in range(num_pads)
        ])

        self.off_ground_ticks = np.zeros(64)
        self.air_time_since_jump = np.zeros(64)

    # Inverted pad lists are reversed, pad i mirrors pad 33 - i
    def get_boost_pads(self, inverted: bool) -> np.ndarray:
        return self.boost_pads[::-1] if inverted else self.boost_pads

    def get_boost_pad_timers(self, inverted: bool) -> np.ndarray:
        return self.boost_pad_timers[::-1] if inverted else self.boost_pad_timers

    def decode(self, packet, ticks_elapsed: int):
        for i in range(len(self.boost_pads)):
            pad = packet.game_boosts[i]
            self.boost_pads[i] = pad.is_active
            # RLBot's timer counts up since pickup, RocketSim's cooldown counts down
            self.boost_pad_timers[i] = 0 if pad.is_active else max(self.pad_cooldowns[i] - pad.timer, 0)

        self.ball = PhysState.from_rlbot(packet.game_ball.physics)
        self.players = [self.decode_player(packet.game_cars[i], i, ticks_elapsed) for i in range(packet.num_cars)]

    def decode_player(self, car, index: int, ticks_elapsed: int) -> Player:
        player = Player()
        player.index = index
        player.car_id = index
        player.team = car.team
        player.phys = PhysState.from_rlbot(car.physics)
        player.boost = car.boost
        player.is_demoed = car.is_demolished
        player.has_jumped = car.jumped

        # TODO: from rlgym-compat, RocketSim's isOnGround has no grace period
        if car.has_wheel_contact:
            self.off_ground_ticks[index] = 0
        else:
            self.off_ground_ticks[index] += ticks_elapsed
        player.is_on_ground = car.has_wheel_contact or self.off_ground_ticks[index] <= 6

        # Same as RocketSim's HasFlipOrJump(), RLBot's double_jumped is set by flips too
        # TODO: RocketSim doesn't count while jump is still held, RLBot doesn't give us that
        if player.is_on_ground or not car.jumped:
            self.air_time_since_jump[index] = 0
        else:
            self.air_time_since_jump[index] += ticks_elapsed * common_values.TICK_TIME
        player.has_flip_or_jump = player.is_on_ground or (
            not car.double_jumped and self.air_time_since_jump[index] < DOUBLEJUMP_MAX_DELAY)

        # RLBot has no world contact, so check if the hitbox touches the floor instead
        # 3uu margin from a parity test vs RocketSim
        rot_z = player.phys.rot_mat[2]
        lowest_z = player.phys.pos[2] + rot_z @ HITBOX_OFFSET - np.abs(rot_z) @ HITBOX_HALF_SIZE
        player.is_turtled = not car.has_wheel_contact and lowest_z < 3

        return player
