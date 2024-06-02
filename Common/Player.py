
import numpy as np
import pyrr
from OpenGL.GL import glUniformMatrix4fv
from OpenGL.GL import GL_FALSE

from Common.Camera import Camera
from Helpers.Globals import bind_globals_to_object, get_global_object_id


class Player:
    global_up = np.array([0, 0, 1], dtype=np.float32)
    __slots__ = 'position', 'theta', 'phi', '_position_matrix', '_angle_matrix', \
                '_view_matrix', 'camera', 'object_id', 'globals'

    def __init__(self, camera: Camera, theta=0, phi=0, position=(0, 0, 0), **kwargs):
        self.camera = camera
        self.position = position
        self.theta = theta
        self.phi = phi

        self.object_id = None
        self.globals = kwargs

        if len(position) != 3:
            raise ValueError(f"Expecting thee cordinates for position, found {len(position)}")
        self.position = np.array(position, dtype=np.float32)

        self._position_matrix = np.identity(4, dtype=np.float32)
        self._angle_matrix = np.identity(4, dtype=np.float32)

        self.recalculate_player_view(position=True, theta=True, phi=True)

    def increment_angles(self, theta=None, phi=None):
        if theta:
            self.theta += theta
            if self.theta > 2 * np.pi or self.theta < -2 * np.pi:
                self.theta = 0

        if phi:
            self.phi += phi
            self.phi = np.clip(self.phi, -0.5 * np.pi, 0.5 * np.pi)

    def set_angles(self, theta=None, phi=None):
        if theta:
            if theta > 2 * np.pi or theta < -2 * np.pi:
                theta = 0
            self.theta = theta

        if phi:
            self.phi = np.clip(phi, -0.5 * np.pi, 0.5 * np.pi)

    def increment_position(self, x=None, y=None, z=None):
        if x:
            self.position[0] += x
        if y:
            self.position[1] += y
        if z:
            self.position[2] += z

    def set_position(self, x=None, y=None, z=None):
        if x:
            self.position[0] = x
        if y:
            self.position[1] = y
        if z:
            self.position[2] = z

    def recalculate_player_view(self, position=False, theta=False, phi=False):
        if position:
            self._position_matrix[3, :3] = self.position

        if theta or phi:
            self._angle_matrix = pyrr.matrix44.create_from_eulers(
                eulers=[self.phi, 0, self.theta],
                dtype=np.float32
            )

        self._view_matrix = self._position_matrix @ self._angle_matrix

    def set_view_to_global(self, shader: int = None, var_name: str = None):
        glob_id = get_global_object_id(self, "object_id", shader, var_name)
        glUniformMatrix4fv(glob_id, 1, GL_FALSE, self._view_matrix)

    def bind_global_variable_names(self, shader):
        bind_globals_to_object(self, shader)
        self.camera.bind_global_variable_names(shader)

    def transform_vertex(self, vertex: np.ndarray):
        result = vertex @ self._view_matrix
        return self.camera.transform_vertex(result)

