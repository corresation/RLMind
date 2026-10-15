import os
import math
import random
import time

import pyrr.vector3

from const import *
from shaders import *
from ribbon import RibbonEmitter

import numpy as np

from pyrr import Quaternion, Matrix33, Matrix44, Vector3

import json

class PhysState:
    def __init__(self):
        self.prev_pos: Vector3 = Vector3((0, 0, 0))
        self.next_pos: Vector3 = Vector3((0, 0, 0))

        self.prev_forward: Vector3 = Vector3((1, 0, 0))
        self.next_forward: Vector3 = Vector3((1, 0, 0))
        self.prev_up: Vector3 = Vector3((0, 0, 1))
        self.next_up: Vector3 = Vector3((0, 0, 1))

        self.has_rot: bool = False
        self.prev_vel: Vector3 = Vector3((0, 0, 0))
        self.next_vel: Vector3 = Vector3((0, 0, 0))
        self.ang_vel: Vector3 = Vector3((0, 0, 0))
        self.has_data: bool = False

    def rotate_with_ang_vel(self, delta_time: float):
        next_right = self.next_up.cross(self.next_forward)
        rot = Matrix33((
            self.next_forward[0], next_right[0], self.next_up[0],
            self.next_forward[1], next_right[1], self.next_up[1],
            self.next_forward[2], next_right[2], self.next_up[2],
        ))

        quat = Quaternion.from_matrix(rot) # type: Quaternion

        # https://stackoverflow.com/questions/24197182/efficient-quaternion-angular-velocity/24201879#24201879
        s_ang_vel = self.ang_vel * delta_time # type: Vector3
        length = s_ang_vel.length
        half = length / 2
        sin = math.sin(half)
        cos = math.cos(half)
        if length > 1e-8:
            rot_quat = Quaternion((s_ang_vel.x * sin, s_ang_vel.y * sin, s_ang_vel.z * sin, length * cos))

            new_quat = rot_quat * quat
            new_rot = Matrix33.from_quaternion(new_quat)

            self.next_forward = Vector3((
                new_rot[0][0],
                new_rot[1][0],
                new_rot[2][0]
            ))
            self.next_up = Vector3((
                new_rot[0][2],
                new_rot[1][2],
                new_rot[2][2]
            ))

    def read_from_json(self, j):
        self.prev_pos = self.next_pos
        self.next_pos = Vector3(j["pos"])

        if not (j.get("forward") is None):
            self.prev_forward = self.next_forward
            self.prev_up = self.next_up
            self.next_forward = Vector3(j["forward"])
            self.next_up = Vector3(j["up"])
            self.has_rot = True
        else:
            self.has_rot = False

        self.prev_vel = self.next_vel
        self.next_vel = Vector3(j["vel"])
        self.ang_vel = Vector3(j["ang_vel"])

        # The first packet has nothing to interpolate from, so don't invent a previous state at the origin
        # (it made new cars count as teleporting and get drawn at 0,0,0 for a packet)
        if not self.has_data:
            self.prev_pos = self.next_pos
            self.prev_forward = self.next_forward
            self.prev_up = self.next_up
            self.prev_vel = self.next_vel
            self.has_data = True

    def is_teleporting(self):
        TELEPORT_DIST_THRESH = 6000 * 0.15
        return (self.prev_pos - self.next_pos).squared_length >= TELEPORT_DIST_THRESH**2

    def get_pos(self, interp_ratio):
        if not self.is_teleporting():
            return self.prev_pos + (self.next_pos - self.prev_pos)*interp_ratio
        else:
            return self.prev_pos

    def get_vel(self, interp_ratio):
        if not self.is_teleporting():
            return self.prev_vel + (self.next_vel - self.prev_vel)*interp_ratio
        else:
            return self.prev_vel

    def get_forward(self, interp_ratio):
        if self.has_rot:
            if not self.is_teleporting():
                return (self.prev_forward + (self.next_forward - self.prev_forward) * interp_ratio).normalized
            else:
                return self.prev_forward
        else:
            return self.next_forward

    def get_up(self, interp_ratio):
        if self.has_rot:
            if not self.is_teleporting():
                return (self.prev_up + (self.next_up - self.prev_up) * interp_ratio).normalized
            else:
                return self.prev_up
        else:
            return self.next_up

class ControllerInputs:
    def __init__(self):
        self.throttle: float = 0
        self.steer: float = 0

        self.pitch: float = 0
        self.yaw: float = 0
        self.roll: float = 0

        self.boost: bool = False
        self.jump: bool = False
        self.handbrake: bool = False

    def read_from_json(self, j):
        self.throttle = j["throttle"]
        self.steer = j["steer"]

        self.pitch = j["pitch"]
        self.yaw = j["yaw"]
        self.roll = j["roll"]

        self.boost = j["boost"]
        self.jump = j["jump"]
        self.handbrake = j["handbrake"]

class CarState:
    def __init__(self):
        self.team_num: int = 0  # Blue = 0, orange = 1

        self.phys: PhysState = PhysState()

        self.controls: ControllerInputs = ControllerInputs()

        self.boost_amount: float = 0  # From 0 to 100
        self.is_boosting = False
        self.on_ground: bool = False
        self.has_flipped_or_double_jumped: bool = False
        self.is_demoed: bool = False
        self.car_id = None
        self.has_flip = None
        self.flip_reset_time = -1
        self.last_update_time = -1

    def read_from_json(self, j, ball_phys=None):
        now = time.time()
        had_flip = self.has_flip
        was_on_ground = self.on_ground
        was_demoed = self.is_demoed
        had_data = self.phys.has_data and self.car_id == j.get("car_id")
        self.car_id = j.get("car_id")
        if not had_data:
            self.phys.has_data = False
        self.team_num = j["team_num"]

        # Respawning after a demo is a teleport from wherever the car died, so start its motion history fresh
        # (otherwise it gets drawn back at the death spot for a packet before jumping to the spawn)
        if self.is_demoed and not j["is_demoed"]:
            self.phys.has_data = False

        self.phys.read_from_json(j["phys"])

        if not (j.get("controls") is None):
            self.controls.read_from_json(j["controls"])

        self.is_boosting = j["boost_amount"] < self.boost_amount
        self.boost_amount = j["boost_amount"]
        self.on_ground = j["on_ground"]
        if not (j.get("has_flipped_or_double_jumped") is None):
            self.has_flipped_or_double_jumped = j["has_flipped_or_double_jumped"]
        self.is_demoed = j["is_demoed"]
        self.has_flip = j.get("has_flip")

        continuous = had_data and not was_demoed and not self.is_demoed and not self.phys.is_teleporting()
        continuous &= 0 <= now - self.last_update_time < 0.5
        if not continuous:
            self.flip_reset_time = -1
        elif ball_phys is not None and not ball_phys.is_teleporting() and self.has_flip is True:
            regained_flip = had_flip is False
            wheel_contact = not was_on_ground and self.on_ground
            if regained_flip or wheel_contact:
                # A floor/wall landing also restores flips. Require the ball against the underside.
                to_ball = ball_phys.next_pos - self.phys.next_pos
                distance = to_ball.length
                if self.phys.next_pos.z > 150 and 1 < distance < 190:
                    underside_align = (-self.phys.next_up).dot(to_ball / distance)
                    if underside_align > 0.5:
                        self.flip_reset_time = now
        self.last_update_time = now

# From RLGym
default_boost_pad_locations = (
    (0, -4240, 70), (-1792, -4184, 70), (1792, -4184, 70), (-3072, -4096, 73), (3072, -4096, 73),
	(- 940, -3308, 70), (940, -3308, 70), (0, -2816, 70), (-3584, -2484, 70), (3584, -2484, 70),
	(-1788, -2300, 70), (1788, -2300, 70), (-2048, -1036, 70), (0, -1024, 70), (2048, -1036, 70),
	(-3584, 0, 73), (-1024, 0, 70), (1024, 0, 70), (3584, 0, 73), (-2048, 1036, 70), (0, 1024, 70),
	(2048, 1036, 70), (-1788, 2300, 70), (1788, 2300, 70), (-3584, 2484, 70), (3584, 2484, 70),
	(0, 2816, 70), (- 940, 3310, 70), (940, 3308, 70), (-3072, 4096, 73), (3072, 4096, 73),
	(-1792, 4184, 70), (1792, 4184, 70), (0, 4240, 70)
)

class RenderState:
    MAX_LINES = 0x2000
    def __init__(self):
        self.lines = []

    def read_from_json(self, j):
        if not (j.get("lines") is None):
            lines = j["lines"]
            if len(lines) > RenderState.MAX_LINES:
                raise Exception(f"Cannot render {len(lines)} lines, maximum is {RenderState.MAX_LINES}")

            for line in lines:
                try:
                    start = Vector3(line["start"])
                    end = Vector3(line["end"])
                except:
                    raise Exception(f"Invalid line format: \"{line}\", format should be [start:[x,y,z], end:[x,y,z]] with two elements")

                self.lines.append([start, end])


class GameState:
    def __init__(self):
        self.ball_state: PhysState = PhysState()
        self.car_states = []

        self.boost_pad_locations = []
        for t in default_boost_pad_locations:
            self.boost_pad_locations.append(Vector3(t))
        self.boost_pad_states = None

        self.recv_time = -1
        self.recv_interval = -1

        self.gamemode = None
        self.render_state = RenderState()

    def is_boost_big(self, idx):
        return self.boost_pad_locations[idx].z == 73

    def read_from_json(self, j):

        self.ball_state.read_from_json(j["ball_phys"])

        j_cars = j["cars"]
        if len(self.car_states) != len(j_cars):
            # Car amount changed, remake cars array
            self.car_states = []
            for j_car in j_cars:
                car_state = CarState()
                car_state.read_from_json(j_car, self.ball_state)
                self.car_states.append(car_state)
        else:
            # Update the cars we already have
            for i in range(len(self.car_states)):
                self.car_states[i].read_from_json(j_cars[i], self.ball_state)

        if not (j.get("boost_pad_locations") is None):
            self.boost_pad_locations = []
            for pos in j.get("boost_pad_locations"):
                self.boost_pad_locations.append(Vector3(pos))

        if not (j.get("boost_pad_states") is None):
            self.boost_pad_states = j.get("boost_pad_states")

            if len(self.boost_pad_states) != len(self.boost_pad_locations):
                raise ("Number of boost pad states ({}) does not match number of boost pad locations ({})"
                       .format(len(self.boost_pad_states), len(self.boost_pad_locations)))
        else:
            self.boost_pad_states = None

        if not (j.get("gamemode") is None):
            self.gamemode = j["gamemode"].lower()
        else:
            self.gamemode = "soccar"

        self.render_state = RenderState()
        if not (j.get("render") is None):
            self.render_state.read_from_json(j["render"])
