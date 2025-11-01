import pygame as pg
from OpenGL.GL import glClearColor, glEnable, glUseProgram, GL_DEPTH_TEST

from Common.ShaderProgram import ShaderProgram


class InspectionEngine:
    """3D Engine that sets up pygame and compiles a single shader."""

    def __init__(self, screen_size, background_col, vertex_shader, fragment_shader):
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(
            pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE
        )
        pg.display.set_mode(screen_size, pg.OPENGL | pg.DOUBLEBUF)
        pg.mouse.set_cursor(*pg.cursors.diamond)
        self.clock = pg.time.Clock()
        glClearColor(*background_col)
        glEnable(GL_DEPTH_TEST)

        self.shader = ShaderProgram.createShader(vertex_shader, fragment_shader)
        glUseProgram(self.shader)
