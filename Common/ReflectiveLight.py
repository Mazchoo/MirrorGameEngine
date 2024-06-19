
import numpy as np
from OpenGL.GL import (glUniform3fv, glUniform1f)

from Helpers.GlobalVarUtil import bind_globals_to_object, get_global_object_id
from Helpers.Globals import LIGHT_FREQUENCY, LIGHT_AMPLITUDE


class ReflectiveLight:
    def __init__(self, position: list, color: list, strength: float,
                 ambient_strength: float, min_dist: float, max_dist: float, **kwargs):

        self.position = np.array(position, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)
        self.strength = float(strength)
        self.min_strength = float(strength)
        self.ambient_strength = float(ambient_strength)
        self.min_dist = float(min_dist)
        self.max_dist = float(max_dist)

        self.globals = kwargs
        self.position_glob_id = None
        self.color_glob_id = None
        self.strength_glob_id = None
        self.ambient_strength_glob_id = None
        self.min_dist_glob_id = None
        self.max_dist_glob_id = None

    def set_position_to_global(self, shader: int = None, var_name: str = None):
        glob_id = get_global_object_id(self, "position_glob_id", shader, var_name)
        glUniform3fv(glob_id, 1, self.position)

    def set_color_to_global(self, shader: int = None, var_name: str = None):
        glob_id = get_global_object_id(self, "color_glob_id", shader, var_name)
        glUniform3fv(glob_id, 1, self.color)

    def set_strength_to_global(self, shader: int = None, var_name: str = None):
        glob_id = get_global_object_id(self, "strength_glob_id", shader, var_name)
        glUniform1f(glob_id, self.strength)

    def set_ambient_strength_to_global(self, shader: int = None, var_name: str = None):
        glob_id = get_global_object_id(self, "ambient_strength_glob_id", shader, var_name)
        glUniform1f(glob_id, self.ambient_strength)

    def set_min_dist_to_global(self, shader: int = None, var_name: str = None):
        glob_id = get_global_object_id(self, "min_dist_glob_id", shader, var_name)
        glUniform1f(glob_id, self.min_dist)

    def set_max_dist_to_global(self, shader: int = None, var_name: str = None):
        glob_id = get_global_object_id(self, "max_dist_glob_id", shader, var_name)
        glUniform1f(glob_id, self.max_dist)

    def set_all_globals(self):
        self.set_position_to_global()
        self.set_color_to_global()
        self.set_strength_to_global()
        self.set_ambient_strength_to_global()
        self.set_min_dist_to_global()
        self.set_max_dist_to_global()

    def bind_global_variable_names(self, shader):
        bind_globals_to_object(self, shader)
    
    def cycle_light_strength(self, frame, shader):
        ''' Add a variational offset to the light strength '''
        self.strength = np.sin(frame * LIGHT_FREQUENCY) * LIGHT_AMPLITUDE + self.min_strength
        self.set_strength_to_global(shader)
