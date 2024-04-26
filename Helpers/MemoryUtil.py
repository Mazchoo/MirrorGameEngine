
from OpenGL.GL import (glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer,
                       glBufferData, glEnableVertexAttribArray, glVertexAttribPointer,
                       GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT, GL_FALSE)
import numpy as np
import ctypes


def generate_vertex_buffers(vertices: np.ndarray):
    # Vertex array object tells us what is in the object
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # We creat a buffer with of data with an id
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    return vao, vbo


def layout_position_texture_normal():
    '''
        Instruct the latest vertex buffer object to read the data
        as x, y, z, s, t, nx, ny, nz
        (x, y, z) position
        (s, t) texture
        (nx, ny, nz) normal
    '''
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))


def layout_position_color():
    '''
        Instruct the latest vertex buffer object to read the data
        as x, y, z, r, g, b
        (x, y, z) position
        (r, g, b) color
    '''
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))


def layout_position_texture():
    '''
        Instruct the latest vertex buffer object to read the data
        as x, y, tx, ty
        (x, y) position
        (tx, ty) texture Cordinate
    '''
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
