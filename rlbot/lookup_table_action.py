import numpy as np

from game_state import Player

# Same as RLGC::LookupTableAction
class LookupTableAction:
    def __init__(self):
        # Boolean input
        R_B = (0, 1)

        R_F = (-1, 0, 1)

        actions = []

        # Ground
        for throttle in R_F:
            for steer in R_F:
                for boost in R_B:
                    for handbrake in R_B:
                        # Prevent useless throttle when boosting
                        if boost == 1 and throttle != 1:
                            continue

                        actions.append([throttle, steer, 0, steer, 0, 0, boost, handbrake])

        num_ground_actions = len(actions)

        # Aerial
        for pitch in R_F:
            for yaw in R_F:
                for roll in R_F:
                    for jump in R_B:
                        for boost in R_B:
                            # Only need roll for sideflip
                            if jump == 1 and yaw != 0:
                                continue

                            # Duplicate with ground
                            if pitch == roll and roll == jump and jump == 0:
                                continue

                            # Enable handbrake for potential wavedashes
                            handbrake = float(jump == 1 and (pitch != 0 or yaw != 0 or roll != 0))

                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])

        self.actions = np.array(actions, dtype=np.float32)

        self.ground_mask = np.zeros(len(actions), dtype=bool)
        self.air_mask = np.zeros(len(actions), dtype=bool)
        self.jump_mask = np.zeros(len(actions), dtype=bool)
        self.boost_mask = np.zeros(len(actions), dtype=bool)

        for i, action in enumerate(self.actions):
            throttle, steer, pitch, yaw, roll, jump, boost, handbrake = action

            if jump:
                self.jump_mask[i] = True

            if boost:
                self.boost_mask[i] = True

            if i < num_ground_actions:
                self.ground_mask[i] = True

            if i >= num_ground_actions and not jump:
                self.air_mask[i] = True

            # Add additional yaw-only actions to air mask
            # These actions were skipped during air action generation to prevent duplicates
            if i < num_ground_actions:
                if throttle == boost and (yaw != 0) == handbrake:
                    self.air_mask[i] = True

    def get_action_amount(self) -> int:
        return len(self.actions)

    def parse_action(self, index: int) -> np.ndarray:
        return self.actions[index]

    def get_action_mask(self, player: Player) -> np.ndarray:
        result = np.zeros(len(self.actions), dtype=bool)

        if player.is_on_ground:
            result |= self.ground_mask
        else:
            result |= self.air_mask

        if player.boost == 0:
            result &= ~self.boost_mask

        if player.has_flip_or_jump or player.is_turtled:
            result |= self.jump_mask

        return result
