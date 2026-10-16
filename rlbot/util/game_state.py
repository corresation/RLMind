import numpy as np
from typing import List

from rlbot.utils.structures.game_data_struct import GameTickPacket, FieldInfoPacket, PlayerInfo

from .physics_object import PhysicsObject
from .player_data import PlayerData
from . import common_values


class GameState:
    def __init__(self, game_info: FieldInfoPacket):
        self.blue_score = 0
        self.orange_score = 0
        self.players: List[PlayerData] = []
        self._on_ground_ticks = np.zeros(64, dtype=np.int32)

        self.ball: PhysicsObject = PhysicsObject()
        self.inverted_ball: PhysicsObject = PhysicsObject()

        n = game_info.num_boosts
        self.boost_pads: np.ndarray = np.zeros(n, dtype=np.float32)
        self.inverted_boost_pads: np.ndarray = np.zeros(n, dtype=np.float32)

        # NEW: countdown timers (seconds until pad becomes available)
        self.boost_pad_timers: np.ndarray = np.zeros(n, dtype=np.float32)
        self.inverted_boost_pad_timers: np.ndarray = np.zeros(n, dtype=np.float32)

        # Respawn time per pad (small=4s, full=10s). Heuristic via Z==73 in CommonValues.
        # This matches the 34 items order used elsewhere.
        # If the runtime map differs, it will still behave reasonably.
        locs = common_values.BOOST_LOCATIONS
        self._respawn = np.array([10.0 if abs(l[2] - 73.0) < 1e-3 else 4.0 for l in locs], dtype=np.float32)

    def get_boost_pads(self, inverted: bool) -> np.ndarray:
        return self.inverted_boost_pads if inverted else self.boost_pads

    def get_boost_pad_timers(self, inverted: bool) -> np.ndarray:
        return self.inverted_boost_pad_timers if inverted else self.boost_pad_timers

    def decode(self, packet: GameTickPacket, ticks_elapsed: int = 1):
        self.blue_score = packet.teams[0].score
        self.orange_score = packet.teams[1].score

        # --- Boost pads + timers ---
        prev_active = self.boost_pads.copy()
        for i in range(packet.num_boost):
            is_active = float(packet.game_boosts[i].is_active)
            self.boost_pads[i] = is_active
            if is_active >= 0.5:
                # Available => timer is 0
                self.boost_pad_timers[i] = 0.0
            else:
                # Just got consumed? reset timer to respawn time; else count down.
                if prev_active[i] >= 0.5:
                    self.boost_pad_timers[i] = self._respawn[i]
                else:
                    dt = ticks_elapsed / 120.0
                    self.boost_pad_timers[i] = max(0.0, self.boost_pad_timers[i] - dt)

        # Invert (index-reversed) view
        self.inverted_boost_pads[:] = self.boost_pads[::-1]
        self.inverted_boost_pad_timers[:] = self.boost_pad_timers[::-1]

        # --- Ball ---
        self.ball.decode_ball_data(packet.game_ball.physics)
        self.inverted_ball.invert(self.ball)

        # --- Players ---
        self.players = []
        for i in range(packet.num_cars):
            player = self._decode_player(packet.game_cars[i], i, ticks_elapsed)
            self.players.append(player)
            if player.ball_touched:
                self.last_touch = player.car_id

    def _decode_player(self, player_info: PlayerInfo, index: int, ticks_elapsed: int) -> PlayerData:
        pd = PlayerData()

        pd.car_data.decode_car_data(player_info.physics)
        pd.inverted_car_data.invert(pd.car_data)

        if player_info.has_wheel_contact:
            self._on_ground_ticks[index] = 0
        else:
            self._on_ground_ticks[index] += ticks_elapsed

        pd.car_id = index
        pd.team_num = player_info.team
        pd.is_demoed = player_info.is_demolished
        pd.on_ground = player_info.has_wheel_contact or self._on_ground_ticks[index] <= 6
        pd.ball_touched = False
        # Keep your previous semantics, but also capture "jumped" explicitly
        pd.has_flip = not player_info.double_jumped
        pd.has_jump = not player_info.double_jumped
        pd.has_jumped = bool(getattr(player_info, "jumped", False))  # RLBot v4 provides .jumped
        pd.boost_amount = player_info.boost / 100.0

        return pd
