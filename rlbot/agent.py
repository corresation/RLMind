import os
import numpy as np
import torch
from discrete_policy import DiscreteFF
from lookup_table_action import LookupTableAction


class Agent:
    def __init__(self):
        self.action_parser = LookupTableAction()
        self.num_actions = len(self.action_parser._lookup_table)
        cur_dir = os.path.dirname(os.path.realpath(__file__))

        device = torch.device("cpu")

        # models.pt from an RLMind checkpoint goes in model/
        save = torch.load(os.path.join(cur_dir, "model", "models.pt"), map_location=device, weights_only=False)
        configs = save["configs"]
        shared_layer_sizes = configs["sharedHead"]["layerSizes"] if "sharedHead" in configs else []
        policy_layer_sizes = configs["policy"]["layerSizes"]

        self.policy = DiscreteFF(
            save["obsSize"], self.num_actions,
            shared_layer_sizes, policy_layer_sizes,
            device
        )

        if "sharedHead" in save["models"]:
            self.policy.shared_head.load_state_dict(save["models"]["sharedHead"])
        self.policy.policy.load_state_dict(save["models"]["policy"])

        torch.set_num_threads(1)

    def act(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            action_idx, _ = self.policy.get_action(state, True)
        action = np.array(self.action_parser.parse_actions([action_idx]))
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
        if action.ndim != 1:
            raise Exception("Invalid action:", action)
        return action
