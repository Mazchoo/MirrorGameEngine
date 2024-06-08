
import numpy as np
from copy import copy

from Common.ObjMtlMesh import ObjMtlMesh
from Common.EulerMotion import EulerMotion
from Helpers.Globals import GRAVITY_CONSTANT, IMAGE_SIZE, CEILING_LEVEL, DESPAWN_LEVEL

class Balloon(ObjMtlMesh):

    __slots__ = 'terminal_velocity', 'velocity', 'density', 'responsive',\
                'running', 'response_count', 'despawn', 'spawn_time', \
                'spawn_count', 'initial_position', 'initial_angles'

    def __init__(self, file_path: str, motion: EulerMotion, normalize_scale: float,
                 drag: float, density: float, spawn_time: int, **kwargs):
        super().__init__(file_path, motion, normalize_scale, **kwargs)

        self.terminal_velocity = drag
        self.velocity = np.array([0, 0, 0], dtype=np.float32)
        self.density = density
        self.responsive = True
        self.running = False
        self.response_count = 0
        self.despawn = False
        self.initial_position = copy(self.motion.position)
        self.initial_angles = copy(self.motion.angles)
        self.spawn_time = spawn_time
        self.spawn_count = spawn_time

    def update(self):
        if self.despawn:
            return

        if self.screen_centroid[1] > DESPAWN_LEVEL + IMAGE_SIZE[1]:
            self.running = False
            self.despawn = True

        if not self.running:
            if self.spawn_count > 0:
                self.spawn_count -= 1
            if self.spawn_count <= 0:
                self.running = True
            return

        if not self.responsive:
            self.response_count -= 1
            if self.response_count < 0:
                self.responsive = True

        self.velocity[1] -= GRAVITY_CONSTANT
        velocity_sq = (self.velocity * self.velocity).sum()
        if velocity_sq > self.terminal_velocity:
            self.velocity *= self.terminal_velocity / velocity_sq

        if self.screen_bbox[2][0] > IMAGE_SIZE[0] and self.velocity[0] < 0:
            self.velocity[0] *= -1

        if self.screen_bbox[0][0] < 0 and self.velocity[0] > 0:
            self.velocity[0] *= -1

        if self.screen_bbox[0][1] < CEILING_LEVEL and self.velocity[1] > 0:
            self.velocity[1] *= -0.5

        self.motion.position += self.velocity
        self.motion.recalculate_motion_matrix(position=True)

    def draw(self):
        if self.despawn or not self.running:
            return
        super().draw()

    def check_collision(self, pose: dict):
        if not self.responsive:
            return

        mass = self.density * self.volume
        body_part_mass = 2 #  ToDo - trying varying this
        min_x, min_y = self.screen_bbox[0]
        max_x, max_y = self.screen_bbox[2]

        for key, body_part_data in pose.items():
            x, y = body_part_data['position']
            if x > min_x and x < max_x and y > min_y and y < max_y:
                collision_v = -1 * body_part_data['velocity']
                collision_v[1] = max(collision_v[1], 0)

                if np.linalg.norm(collision_v) < 1e-7:
                    continue

                velocity = (mass * self.velocity[:2] + body_part_mass * collision_v)
                velocity /= (mass + body_part_mass)
                self.velocity[:2] = velocity

                self.response_count = 5
                self.responsive = False
                break

    def respawn(self):
        self.motion.position = self.initial_position
        self.motion.angles = self.initial_angles
        self.velocity = np.array([0, 0, 0], dtype=np.float32)
        self.spawn_count = self.spawn_time
        self.despawn = True 
