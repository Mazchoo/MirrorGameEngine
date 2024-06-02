
import numpy as np

from Common.ObjMtlMesh import ObjMtlMesh
from Common.EulerMotion import EulerMotion
from PoseEstimation.pose import Pose
from Helpers.Globals import GRAVITY_CONSTANT, IMAGE_SIZE

class Balloon(ObjMtlMesh):

    __slots__ = 'drag', 'velocity', 'density', 'responsive', 'running', 'response_count'

    def __init__(self, file_path: str, motion: EulerMotion, normalize_scale: float,
                 drag: float, density: float, **kwargs):
        super().__init__(file_path, motion, normalize_scale, **kwargs)

        self.drag = drag
        self.velocity = np.array([0, 0, 0], dtype=np.float32)
        self.density = density
        self.responsive = True
        self.running = True
        self.response_count = 0

    def update(self):
        if not self.responsive:
            self.response_count -= 1
            if self.response_count < 0:
                self.responsive = True

        if not self.running:
            return

        self.velocity[1] -= GRAVITY_CONSTANT
        velocity_sq = (self.velocity * self.velocity).sum()
        if velocity_sq > self.drag:
            self.velocity *= self.drag / velocity_sq

        self.motion.position += self.velocity
        self.motion.recalculate_motion_matrix(position=True)

    def check_collision(self, pose: dict):
        if not self.responsive:
            return

        mass = self.density * self.volume
        body_part_mass = 0.5 #  ToDo - trying varying this
        min_x, min_y = self.screen_bbox[0]
        max_x, max_y = self.screen_bbox[2]

        for body_part_data in pose.values():
            x, y = body_part_data['position']
            if x > min_x and x > max_x and y > min_y and y < max_y:
                collision_v = -1 * body_part_data['velocity']

                if np.linalg.norm(collision_v) < 1e-7:
                    continue

                velocity = (mass * self.velocity[:2] + body_part_mass * collision_v)
                velocity /= (mass + body_part_mass)
                self.velocity[:2] = velocity

                self.response_count = 30
                self.responsive = False
                break

        
