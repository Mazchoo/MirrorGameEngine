from OpenGL.GL import (
    glBindVertexArray,
    glDrawArrays,
    glDeleteVertexArrays,
    glDeleteBuffers,
)
from OpenGL.GL import GL_TRIANGLES
from Common.DynamicMaterial import DynamicMaterial
from Helpers.Globals import IMAGE_VERTICES
import numpy as np
from Helpers.MemoryUtil import generate_vertex_buffers, layout_position_texture


class DynamicOverlay:
    __slots__ = "material", "vao", "vbo"

    def __init__(self):
        vertices = np.array(IMAGE_VERTICES, dtype=np.float32)
        self.vao, self.vbo = generate_vertex_buffers(vertices)
        layout_position_texture()
        self.material = DynamicMaterial()

    def setTexture(self, image: np.ndarray):
        self.material.setTextureFromImage(image)

    def draw(self):
        self.material.use()
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

    def destroy(self):
        self.material.destroy()
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))
