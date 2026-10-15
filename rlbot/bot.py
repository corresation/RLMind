import glob
import json
import os
import numpy as np
import torch
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from torch import nn

import common_values
from lookup_table_action import LookupTableAction
from game_state import GameState
from obs import AdvancedObs, AdvancedObsPadded

ACTIVATIONS = {"relu": nn.ReLU, "leakyRelu": nn.LeakyReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}

def find_model_folder() -> str:
    here = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(os.path.join(here, "bot", "models.pt")):
        return os.path.join(here, "bot")

    saves = glob.glob(os.path.join(here, "..", "out", "build", "*", "checkpoints", "*", "models.pt"))
    assert saves, "No model in bot/ and no checkpoints to use"
    return os.path.dirname(max(saves, key=os.path.getmtime))

def make_model(num_inputs: int, cfg: dict, num_outputs: int = 0) -> nn.Sequential:
    layers = []
    for size in cfg["layerSizes"]:
        layers.append(nn.Linear(num_inputs, size))
        if cfg["layerNorm"]:
            layers.append(nn.LayerNorm(size))
        layers.append(ACTIVATIONS[cfg["activation"]]())
        num_inputs = size
    if num_outputs:
        layers.append(nn.Linear(num_inputs, num_outputs))
    return nn.Sequential(*layers)

def loadPolicy(folder: str):
    state = torch.load(os.path.join(folder, "models.pt"), map_location="cpu", weights_only=False)
    with open(os.path.join(folder, "stats.json")) as f:
        stats = json.load(f)

    configs = state["configs"]
    head, head_size = None, state["obsSize"]
    if "sharedHead" in configs:
        head = make_model(head_size, configs["sharedHead"])
        head.load_state_dict(state["models"]["sharedHead"])
        head_size = configs["sharedHead"]["layerSizes"][-1]
    policy = make_model(head_size, configs["policy"], state["numActions"])
    policy.load_state_dict(state["models"]["policy"])

    config = {"tickSkip": stats["tickSkip"], "actionDelay": stats["actionDelay"],
              "obsSize": state["obsSize"], "numActions": state["numActions"]}
    return config, head, policy

def maskedProbs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits - 1e10 * ~mask, -1)

class Bot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        folder = find_model_folder()
        config, self.head, self.policy = loadPolicy(folder)
        self.tickSkip, self.actionDelay = config["tickSkip"], config["actionDelay"]
        self.obsSize = config["obsSize"]
        self.deterministic = True
        assert 0 <= self.actionDelay < self.tickSkip

        # A padded model always takes the 3v3 size, so it works in any mode
        self.obs_builder = AdvancedObsPadded() if self.obsSize == AdvancedObsPadded.OBS_SIZE else AdvancedObs()
        self.action_parser = LookupTableAction()
        assert self.action_parser.get_action_amount() == config["numActions"]

        torch.set_num_threads(1)

    @torch.no_grad()
    def infer_action(self, t_obs, t_mask, deterministic):
        logits = self.policy(self.head(t_obs) if self.head else t_obs)
        probs = maskedProbs(logits, t_mask)
        return int(probs.argmax()) if deterministic else int(torch.multinomial(probs, 1))

    def initialize_agent(self):
        self.game_state = GameState(self.get_field_info())
        self.controls = SimpleControllerState()
        self.action = np.zeros(common_values.NUM_ACTIONS)
        self.prev_action = np.zeros(common_values.NUM_ACTIONS)
        self.ticks = self.tickSkip
        self.update_action = True
        self.prev_time = None

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = packet.game_info.seconds_elapsed
        ticks_elapsed = 1 if self.prev_time is None else round((cur_time - self.prev_time) * common_values.TICK_RATE)
        self.prev_time = cur_time

        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

        # Same timing as Nexto's RLBot bot, which actionDelay = tickSkip - 1 is based on
        if self.update_action:
            self.update_action = False

            player = self.game_state.players[self.index]
            obs = self.obs_builder.build_obs(player, self.game_state, self.prev_action)
            assert len(obs) == self.obsSize, f"Obs is {len(obs)} long, the model wants {self.obsSize} (wrong game mode?)"
            t_obs = torch.as_tensor(obs)
            t_mask = torch.as_tensor(self.action_parser.get_action_mask(player))
            self.action = self.action_parser.parse_action(self.infer_action(t_obs, t_mask, self.deterministic))

            self.prev_action = self.action

        if self.ticks >= self.actionDelay:
            self.update_controls(self.action)

        if self.ticks >= self.tickSkip:
            self.ticks = 0
            self.update_action = True

        return self.controls

    def update_controls(self, action: np.ndarray):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0
