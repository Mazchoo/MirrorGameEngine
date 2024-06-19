from numbers import Number

import numpy as np
from OpenGL.GL import (glBindTexture, glGenTextures, glTexParameter, glTexImage2D,
                       glGenerateMipmap, glGetUniformLocation, glUniform3fv,
                       glUniform1f, glActiveTexture, glDeleteTextures)
from OpenGL.GL import (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_REPEAT,
                       GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_NEAREST, GL_LINEAR,
                       GL_RGBA, GL_UNSIGNED_BYTE, GL_TEXTURE0)

from Helpers.ImageUtil import read_image_data_from_source
from Helpers.Globals import (SPECULAR_EXPONENT_WEIGHTING, MIN_SPECULAR_EXPONENT, DEFAULT_AMBIENT_WEIGHTING,
                             DEFAULT_DIFFUSE_WEIGHTING, DEFAULT_SPECULAR_WEIGHTING, DEFAULT_SPECULAR_EXPONENT,
                             DEFAULT_OPACITY, DEFAULT_SPECULAR_TINT)


GLOBAL_ID_NAMES = (
    'ambient_weighting_glob_id',
    'diffuse_weighting_glob_id',
    'specular_weighting_glob_id',
    'specular_exponent_glob_id',
    'opacicty_glob_id',
    'specular_tint_glob_id',
)


def get_value_or_default(query_dict: dict, key: str, default_value):
    ''' Get value from parameters provided or return default '''

    result = query_dict[key] if key in query_dict else default_value

    if isinstance(result, list) or isinstance(result, tuple):
        result = np.array(result, dtype=np.float32)
    elif isinstance(result, Number):
        result = float(result)

    return result


class ShinyMaterial:
    '''
        A material with defined reflective properties.
        Should handle all output produced by Helper.ReadMtl
    '''

    def __init__(self, hue_offset, **kwargs):
        # ToDo : Consider making a class with a struct for global variables
        self.ambient_weighting = get_value_or_default(kwargs, 'ambient_weighting', DEFAULT_AMBIENT_WEIGHTING)
        self.diffuse_weighting = get_value_or_default(kwargs, 'diffuse_weighting', DEFAULT_DIFFUSE_WEIGHTING)
        self.specular_weighting = get_value_or_default(kwargs, 'specular_weighting', DEFAULT_SPECULAR_WEIGHTING)
        self.specular_exponent = get_value_or_default(kwargs, 'specular_exponent', DEFAULT_SPECULAR_EXPONENT)
        self.specular_exponent = max(self.specular_exponent * SPECULAR_EXPONENT_WEIGHTING, MIN_SPECULAR_EXPONENT)
        self.opacicty = get_value_or_default(kwargs, 'opacicty', DEFAULT_OPACITY)
        self.specular_tint = get_value_or_default(kwargs, 'specular_tint', DEFAULT_SPECULAR_TINT)

        for global_id_name in GLOBAL_ID_NAMES:
            self.__setattr__(global_id_name, None)

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        image_data, width, height = read_image_data_from_source(kwargs["texture"], hue_offset)

        # First argument is used to downscale image at further distances
        # Other zero argument represents border color
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        self.global_check_passed = False

    def assign_global_slots(self, shader, **kwargs):
        for global_id_name in GLOBAL_ID_NAMES:
            if global_name := get_value_or_default(kwargs, global_id_name, None):
                self.__setattr__(global_id_name, glGetUniformLocation(shader, global_name))

    def set_material_to_global(self):
        if not self.global_check_passed:
            for global_id_name in GLOBAL_ID_NAMES:
                if self.__getattribute__(global_id_name) is None:
                    raise AttributeError("Object Id has not been set.")
            self.global_check_passed = True

        glUniform3fv(self.ambient_weighting_glob_id, 1, self.ambient_weighting)
        glUniform3fv(self.diffuse_weighting_glob_id, 1, self.diffuse_weighting)
        glUniform3fv(self.specular_weighting_glob_id, 1, self.specular_weighting)
        glUniform1f(self.specular_exponent_glob_id, self.specular_exponent)
        glUniform1f(self.opacicty_glob_id, self.opacicty)
        glUniform1f(self.specular_tint_glob_id, self.specular_tint)

    def use(self, slot=GL_TEXTURE0):
        self.set_material_to_global()
        glActiveTexture(slot)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def destroy(self):
        glDeleteTextures(1, (self.texture, ))
