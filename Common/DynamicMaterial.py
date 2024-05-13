
from OpenGL.GL import (glBindTexture, glTexParameter, glTexImage2D,
                       glGenerateMipmap, glActiveTexture, glDeleteTextures,
                       glGenTextures)
from OpenGL.GL import (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
                       GL_REPEAT, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
                       GL_NEAREST, GL_LINEAR, GL_RGB, GL_UNSIGNED_BYTE, GL_TEXTURE1)

from pycuda.gl import graphics_map_flags, RegisteredImage
import pycuda.driver as cuda
import numpy as np
import torch


class DynamicMaterial:

    __slots__ = 'texture' # , 'cuda_texture'

    def __init__(self):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        # self.cuda_texture = RegisteredImage(self.texture, GL_TEXTURE_2D, graphics_map_flags.WRITE_DISCARD)

        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def setTextureFromImage(self, image: np.ndarray):
        glBindTexture(GL_TEXTURE_2D, self.texture)

        height, width, _ = image.shape

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image)
        glGenerateMipmap(GL_TEXTURE_2D)

    def setTextureFromTorchArray(self, t: torch.Tensor):
        ''' TODO test this. '''
        self.cuda_texture.map()
        array, _ = self.cuda_texture.get_mapped_array()
        cuda.memcpy_dtod(array, t.data_ptr(), t.numel() * t.element_size())
        self.cuda_texture.unmap()

    def use(self, slot=GL_TEXTURE1):
        glActiveTexture(slot)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def destroy(self):
        glDeleteTextures(1, (self.texture, ))
