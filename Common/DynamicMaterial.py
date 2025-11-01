from OpenGL.GL import (
    glBindTexture,
    glTexParameter,
    glTexImage2D,
    glGenerateMipmap,
    glActiveTexture,
    glDeleteTextures,
    glGenTextures,
)
from OpenGL.GL import (
    GL_TEXTURE_2D,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_REPEAT,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER,
    GL_NEAREST,
    GL_LINEAR,
    GL_RGB,
    GL_UNSIGNED_BYTE,
    GL_TEXTURE1,
)

import numpy as np


class DynamicMaterial:
    __slots__ = "texture"

    def __init__(self):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, None
        )
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def setTextureFromImage(self, image: np.ndarray):
        glBindTexture(GL_TEXTURE_2D, self.texture)

        height, width, _ = image.shape

        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image
        )
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self, slot=GL_TEXTURE1):
        glActiveTexture(slot)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def destroy(self):
        glDeleteTextures(1, (self.texture,))
