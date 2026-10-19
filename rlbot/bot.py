import numpy as np
import os
import json
import torch

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util import common_values
from util.game_state import GameState
from util.player_data import PlayerData

from agent import Agent as Agent_Base
from obs import AdvancedObs

# in case kickoff model but in this case none
import sys

import math

class ExampleBot(BaseAgent): # Skidded from LittleThing so sorry
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        # main model
        #print("Initializing main model...")
        self.base_agent = Agent_Base()
        self.base_obs_builder = AdvancedObs()

        # model switching state
        self.current_model = "main"  # "main" or "kickoff"
        self.model_switch_time = 0.0
        self.kickoff_completed = True

        # normalization logic
        self.obs_mean = None
        self.obs_std = None


        # config model tick skip
        self.main_tick_skip = 8      # main model uses tick_skip = 8
        self.kickoff_tick_skip = 4   # Kickoff model uses tick_skip = 4 for example
        self.tick_skip = self.main_tick_skip  # active tick_skip
        
        self.game_state: GameState = None
        self.controls = SimpleControllerState()
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0

    def is_hot_reload_enabled(self):
        return True

    def initialize_agent(self, field_info=None):
        if field_info is None:
            field_info = type('FieldInfo', (), {'num_boosts': 34})()
        self.field_info = field_info
        self.game_state = GameState(self.field_info)
        self.update_action = True
        self.tick_skip = self.main_tick_skip  # main model tick_skip
        self.ticks = self.tick_skip
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.current_model = "main"
        self.kickoff_completed = False

    def should_use_kickoff_model(self, packet): #in this case return false logic not needed
        return False
        """Determine if we should use the kickoff model"""
        # use kickoff model during kickoff pause
        if packet.game_info.is_kickoff_pause:
            return True
        
        # kickoff model for a short time after kickoff ends to complete any ongoing actions
        if self.current_model == "kickoff" and not packet.game_info.is_kickoff_pause:
            time_since_kickoff = packet.game_info.seconds_elapsed - self.model_switch_time
            if time_since_kickoff < 2.0:  # 2 second grace period
                return True
        
        return False

    def switch_to_kickoff_model(self, packet):
        """Switch to kickoff model"""
        if self.current_model != "kickoff":
            #print(f"[bot] Switching to KICKOFF model at {packet.game_info.seconds_elapsed:.2f}s")
            #print(f"[bot] Changing tick_skip from {self.tick_skip} to {self.kickoff_tick_skip}")
            self.current_model = "kickoff"
            self.model_switch_time = packet.game_info.seconds_elapsed
            self.kickoff_completed = False
            # tick_skip for kickoff model
            self.tick_skip = self.kickoff_tick_skip

    def switch_to_main_model(self, packet):
        """Switch back to main model"""
        if self.current_model != "main":
            #print(f"[bot] Switching to MAIN model at {packet.game_info.seconds_elapsed:.2f}s")
            #print(f"[bot] Changing tick_skip from {self.tick_skip} to {self.main_tick_skip}")
            self.current_model = "main"
            self.model_switch_time = packet.game_info.seconds_elapsed
            self.kickoff_completed = True
            # tick_skip for main model
            self.tick_skip = self.main_tick_skip

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        ticks_elapsed = round(delta * 120)
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

        # switching logic
        if self.should_use_kickoff_model(packet):
            self.switch_to_kickoff_model(packet)
        else:
            self.switch_to_main_model(packet)

        # action update logic
        if self.update_action and len(self.game_state.players) > self.index:
            self.update_action = False
            player = self.game_state.players[self.index]
            teammates = [p for p in self.game_state.players if p.team_num == self.team and p != player]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]
            self.game_state.players = [player] + teammates + opponents

            # which model to use based on current state, the kickoff model is unused
            if self.current_model == "kickoff":
                # kickoff model
                obs = self.kickoff_obs_builder.build_obs(player, self.game_state, self.action)
                if self.obs_mean is not None and self.obs_std is not None:
                    obs = (obs - self.obs_mean) / self.obs_std
                self.action = self.kickoff_agent.act(obs)
                #print(f"[bot] Using KICKOFF model (tick_skip={self.tick_skip}) - Action: {self.action}")
            else:
                # main model
                obs = self.base_obs_builder.build_obs(player, self.game_state, self.action)
                if self.obs_mean is not None and self.obs_std is not None:
                    obs = (obs - self.obs_mean) / self.obs_std
                self.action = self.base_agent.act(obs)
                #print(f"[bot] Using MAIN model (tick_skip={self.tick_skip}) - Action: {self.action}")

        if self.ticks >= self.tick_skip - 1:
            self.update_controls(self.action)

        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_action = True


        return self.controls

    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0