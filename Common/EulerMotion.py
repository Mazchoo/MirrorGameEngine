
import pyrr
import numpy as np
from OpenGL.GL import glUniformMatrix4fv
from OpenGL.GL import GL_FALSE

from Helpers.GlobalVarUtil import bind_globals_to_object, get_global_object_id


class EulerMotion:
    __slots__ = 'angles', 'position', '_angle_matrix', '_position_matrix', \
                'object_id', 'motion_matrix', "globals"

    def __init__(self, position: list, angles: list, **kwargs):

        self.object_id = None
        self.globals = kwargs

        if len(angles) != 3:
            raise ValueError(f"Expecting three angles, found {len(angles)}")
        self.angles = angles

        if len(position) != 3:
            raise ValueError(f"Expecting thee cordinates for position, found {len(position)}")
        self.position = np.array(position, dtype=np.float32)

        self._angle_matrix = np.identity(4, dtype=np.float32)
        self._position_matrix = np.identity(4, dtype=np.float32)

        self.recalculate_motion_matrix(True, True)

    def increment_angles(self, xy=None, yz=None, xz=None):
        if yz:
            self.angles[0] += yz
            if self.angles[0] > 2 * np.pi or self.angles[0] < -2 * np.pi:
                self.angles[0] = 0
        if xy:
            self.angles[1] += xy
            if self.angles[1] > 2 * np.pi or self.angles[1] < -2 * np.pi:
                self.angles[1] = 0
        if xz:
            self.angles[2] += xz
            if self.angles[2] > 2 * np.pi or self.angles[2] < -2 * np.pi:
                self.angles[2] = 0

    def increment_position(self, x=None, y=None, z=None):
        if x:
            self.position[0] += x
        if y:
            self.position[1] += y
        if z:
            self.position[2] += z

    def recalculate_motion_matrix(self, position=False, angles=False):
        if position:
            self._position_matrix[3, :3] = self.position

        if angles:
            self._angle_matrix = pyrr.matrix44.create_from_eulers(
                eulers=self.angles,
                dtype=np.float32
            )

        self.motion_matrix = self._angle_matrix @ self._position_matrix
        return self.motion_matrix

    def set_motion_to_global(self, shader: int = None, var_name: str = None):
        glob_id = get_global_object_id(self, "object_id", shader, var_name)
        glUniformMatrix4fv(glob_id, 1, GL_FALSE, self.motion_matrix)

    def bind_global_variable_names(self, shader: int):
        bind_globals_to_object(self, shader)

    def transform_vertex(self, vertex: np.ndarray) -> np.ndarray:
        ''' Transform 4d vertex object's position in 4d '''
        result = vertex @ self._position_matrix
        return result
