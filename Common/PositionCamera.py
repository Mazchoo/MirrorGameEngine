import numpy as np
from OpenGL.GL import glUniform3fv

from Common.Camera import Camera
from Helpers.GlobalVarUtil import bind_globals_to_object, get_global_object_id


class PositionCamera(Camera):
    """
    Camera that has a position to to be used directly
    in the shader.
    """

    __slots__ = "position_glob_id", "position", "globals"

    def __init__(self, fovy, aspect, near, far, position, **kwargs):
        super().__init__(fovy, aspect, near, far)
        self.position_glob_id = None
        self.position = np.array(position, dtype=np.float32)
        self.globals = kwargs

    def set_position_to_global(self, shader: int = None, var_name: str = None):
        glob_id = get_global_object_id(self, "position_glob_id", shader, var_name)
        glUniform3fv(glob_id, 1, self.position)

    def bind_global_variable_names(self, shader):
        bind_globals_to_object(self, shader)
