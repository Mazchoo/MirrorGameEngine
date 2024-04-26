
import pyrr
import numpy as np
from OpenGL.GL import glUniformMatrix4fv
from OpenGL.GL import GL_FALSE

from Helpers.Globals import bind_globals_to_object, get_global_object_id


class Camera:
    __slots__ = 'fovy', 'aspect', 'near', 'far', 'object_id', 'projection_matrix', 'globals'

    def __init__(self, fovy, aspect, near, far, **kwargs):
        self.fovy = fovy
        self.aspect = aspect
        self.near = near
        self.far = far

        self.globals = kwargs
        self.object_id = None

        self.recalculate_projection(fovy, aspect, near, far)

    def recalculate_projection(self, fovy=None, aspect=None, near=None, far=None):
        fovy = fovy or self.fovy
        aspect = aspect or self.aspect
        near = near or self.near
        far = far or self.far

        self.projection_matrix = pyrr.matrix44.create_perspective_projection(
            fovy=fovy, aspect=aspect, near=near, far=far, dtype=np.float32
        )

    def set_projection_to_global(self, shader: int = None, var_name: str = None):
        glob_id = get_global_object_id(self, "object_id", shader, var_name)
        glUniformMatrix4fv(glob_id, 1, GL_FALSE, self.projection_matrix)

    def bind_global_variable_names(self, shader):
        bind_globals_to_object(self, shader)
