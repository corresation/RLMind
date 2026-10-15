import math
import numpy as np

INV_VEC = np.array([-1, -1, 1])

def rot_mat_from_euler(pitch: float, yaw: float, roll: float) -> np.ndarray:
    # Columns are forward, right, up, same as RocketSim's Angle::ToRotMat()
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array([
        [cp * cy, cy * sp * sr - cr * sy, -cr * cy * sp - sr * sy],
        [cp * sy, sy * sp * sr + cr * cy, -cr * sy * sp + sr * cy],
        [sp, -cp * sr, cp * cr],
    ])

class PhysState:
    def __init__(self, pos=None, rot_mat=None, vel=None, ang_vel=None):
        self.pos = np.zeros(3) if pos is None else pos
        self.rot_mat = np.eye(3) if rot_mat is None else rot_mat
        self.vel = np.zeros(3) if vel is None else vel
        self.ang_vel = np.zeros(3) if ang_vel is None else ang_vel

    @property
    def forward(self) -> np.ndarray:
        return self.rot_mat[:, 0]

    @property
    def up(self) -> np.ndarray:
        return self.rot_mat[:, 2]

    @staticmethod
    def from_rlbot(physics) -> "PhysState":
        rot = physics.rotation
        return PhysState(
            np.array([physics.location.x, physics.location.y, physics.location.z]),
            rot_mat_from_euler(rot.pitch, rot.yaw, rot.roll),
            np.array([physics.velocity.x, physics.velocity.y, physics.velocity.z]),
            np.array([physics.angular_velocity.x, physics.angular_velocity.y, physics.angular_velocity.z]),
        )

    # Same as RLGC::InvertPhys()
    def invert(self) -> "PhysState":
        return PhysState(self.pos * INV_VEC, self.rot_mat * INV_VEC[:, None], self.vel * INV_VEC, self.ang_vel * INV_VEC)
