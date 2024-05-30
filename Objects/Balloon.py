
import numpy as np

from Common.ObjMtlMesh import ObjMtlMesh
from Common.EulerMotion import EulerMotion
from Helpers.Globals import GRAVITY_CONSTANT

class Balloon(ObjMtlMesh):

    __slots__ = 'drag', 'velocity', 

    def __init__(self, file_path: str, motion: EulerMotion, normalize_scale: float,
                 drag: float, **kwargs):
        super().__init__(file_path, motion, normalize_scale, **kwargs)

        self.drag = drag
        self.velocity = np.array([0, 0, 0], dtype=np.float32)

    def update(self):
        self.velocity[1] -= GRAVITY_CONSTANT
        velocity_sq = (self.velocity * self.velocity).sum()
        drag = self.drag * np.sign(self.velocity) * velocity_sq
        drag_overpowered_velcocity = np.abs(self.velocity) < np.abs(drag)
        for i in range(3):
            if drag_overpowered_velcocity[i]:
                self.velocity[i] = 0
            else:
                self.velocity[i] -= drag[i]

        self.motion.position += self.velocity
        self.motion.recalculate_motion_matrix(position=True)
