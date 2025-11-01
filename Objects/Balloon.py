from copy import copy

import numpy as np

from Common.ObjMtlMesh import ObjMtlMesh
from Common.EulerMotion import EulerMotion
from Helpers.Globals import (
    GRAVITY_CONSTANT,
    IMAGE_SIZE,
    CEILING_LEVEL,
    DESPAWN_LEVEL,
    WALL_MOMENTUM_DROP,
    CEILING_MOMENTUM_DROP,
    FRAME_YZ_SPEED_DROP,
    FRAME_XY_DISTURBANCE_RATIO,
    FRAME_XZ_SPEED_DROP,
)


class Balloon(ObjMtlMesh):
    __slots__ = (
        "terminal_velocity",
        "velocity",
        "angular_velocity",
        "density",
        "responsive",
        "running",
        "response_count",
        "despawn",
        "spawn_time",
        "spawn_count",
        "initial_position",
        "initial_angles",
    )

    def __init__(
        self,
        file_path: str,
        motion: EulerMotion,
        normalize_scale: float,
        color_variation: dict,
        terminal_velocity: float,
        density: float,
        spawn_time: int,
        **kwargs,
    ):
        super().__init__(file_path, motion, normalize_scale, color_variation, **kwargs)

        # Velocities
        self.velocity = np.array([0, 0, 0], dtype=np.float32)
        self.angular_velocity = np.array([0, 0, 0], dtype=np.float32)

        # Constants
        self.density = density
        self.spawn_time = spawn_time
        self.spawn_count = spawn_time
        self.terminal_velocity = terminal_velocity
        self.initial_position = self.motion.position.copy()
        self.initial_angles = self.motion.angles.copy()

        # Running state
        self.responsive = True
        self.running = False
        self.response_count = 0
        self.despawn = False

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
            self.velocity[0] *= -WALL_MOMENTUM_DROP

        if self.screen_bbox[0][0] < 0 and self.velocity[0] > 0:
            self.velocity[0] *= -WALL_MOMENTUM_DROP

        if self.screen_bbox[0][1] < CEILING_LEVEL and self.velocity[1] > 0:
            self.velocity[1] *= -CEILING_MOMENTUM_DROP

        self.angular_velocity[0] *= FRAME_YZ_SPEED_DROP
        self.angular_velocity[1] -= FRAME_XY_DISTURBANCE_RATIO * self.motion.angles[1]
        self.angular_velocity[2] *= FRAME_XZ_SPEED_DROP

        self.motion.position += self.velocity
        self.motion.angles += self.angular_velocity
        self.motion.recalculate_motion_matrix(position=True, angles=True)

    def draw(self):
        if self.despawn or not self.running:
            return
        super().draw()

    def check_player_collision(self, pose: dict) -> bool:
        if not self.responsive:
            return False

        mass = self.density * self.volume
        body_part_mass = 3  #  ToDo - trying varying this
        min_x, min_y = self.screen_bbox[0]
        max_x, max_y = self.screen_bbox[2]

        for key, body_part_data in pose.items():
            x, y = body_part_data["position"]
            if x > min_x and x < max_x and y > min_y and y < max_y:
                collision_v = -1 * body_part_data["velocity"]
                collision_v[1] = max(collision_v[1], 0)

                if np.linalg.norm(collision_v) < 1e-7:
                    continue

                velocity = mass * self.velocity[:2] + body_part_mass * collision_v
                velocity /= mass + body_part_mass
                self.velocity[:2] = velocity

                self.angular_velocity[0] += 0.05
                self.angular_velocity[1] -= 0.0005 * collision_v[0]
                self.angular_velocity[2] += 0.02

                self.response_count = 5
                self.responsive = False
                return True

        return False

    @staticmethod
    def check_balloon_collision(balloons: list) -> bool:
        if len(balloons) < 2:
            return False

        balloon = balloons[0]

        mass = balloon.density * balloon.volume
        min_x, min_y = balloon.screen_bbox[0]
        max_x, max_y = balloon.screen_bbox[2]

        for other in balloons[1:]:
            other_mass = other.density * other.volume
            other_min_x, other_min_y = other.screen_bbox[0]
            other_max_x, other_max_y = other.screen_bbox[2]

            if min_x > other_max_x or other_min_x > max_x:
                continue

            if min_y > other_max_y or other_min_y > max_y:
                continue

            speed = mass * balloon.velocity[0] + other_mass * other.velocity[0]
            speed /= mass + other_mass
            speed = max(speed, 0.01)

            if balloon.screen_centroid[0] < other.screen_centroid[0]:
                balloon.velocity[0] = speed
                other.velocity[0] = -speed
            else:
                balloon.velocity[0] = -speed
                other.velocity[0] = speed

            return True

        return False

    def respawn(self):
        self.motion.position = self.initial_position.copy()
        self.motion.angles = self.initial_angles.copy()
        self.motion.recalculate_motion_matrix(position=True, angles=True)

        self.velocity = np.array([0, 0, 0], dtype=np.float32)
        self.angular_velocity = np.array([0, 0, 0], dtype=np.float32)

        self.spawn_count = self.spawn_time
        self.despawn = False
        self.running = False
