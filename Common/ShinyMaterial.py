
from OpenGL.GL import (glBindTexture, glGenTextures, glTexParameter, glTexImage2D,
                       glGenerateMipmap, glGetUniformLocation, glUniform3fv,
                       glUniform1f, glActiveTexture, glDeleteTextures)
from OpenGL.GL import (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_REPEAT,
                       GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_NEAREST, GL_LINEAR,
                       GL_RGBA, GL_UNSIGNED_BYTE, GL_TEXTURE0)
import pygame as pg
import numpy as np
from numbers import Number

# This value determines the shinyness of a material
# Values in mtl files range from 0-400
# The exponent should be 0.1 (really shiny)
# and it can be more than one
SPECULAR_EXPONENT_WEIGHTING = 1 / 255.
MIN_SPECULAR_EXPONENT = 0.2

DEFAULT_AMBIENT_WEIGHTING = (0.2, 0.2, 0.2)
DEFAULT_DIFFUSE_WEIGHTING = (0.5, 0.5, 0.5)
DEFAULT_SPECULAR_WEIGHTING = (1, 1, 1)
DEFAULT_SPECULAR_EXPONENT = 0.75
DEFAULT_OPACITY = 1.
DEFAULT_SPECULAR_TINT = 0.


GLOBAL_ID_NAMES = (
    'ambient_weighting_glob_id',
    'diffuse_weighting_glob_id',
    'specular_weighting_glob_id',
    'specular_exponent_glob_id',
    'opacicty_glob_id',
    'specular_tint_glob_id',
)


def get_value_or_default(query_dict: dict, key: str, default_value):

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

    def __init__(self, **kwargs):
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

        # ToDo : Make image util for this
        texture_source = kwargs['texture']
        if isinstance(texture_source, str):
            image = pg.image.load(texture_source).convert_alpha()
        else:
            texture_array = np.array([[texture_source]], dtype=np.float32)
            image = pg.surfarray.make_surface(texture_array)

        image_width, image_height = image.get_rect().size
        image_data = pg.image.tostring(image, "RGBA")

        # First argument is used to downscale image at further distances
        # Other zero argument represents border color
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
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