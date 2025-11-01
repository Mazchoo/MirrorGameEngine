import pygame as pg
from OpenGL.GL import glClearColor, glEnable
from OpenGL.GL import GL_DEPTH_TEST

from Common.ShaderProgram import ShaderProgram


class OverlayEngine:
    """
    3D Engine that sets up pygame and compiles multiple shader.
    Can switch between shaders when in a game loop.
    """

    __slots__ = "_shaders", "_current_id", "clock"

    def __init__(self, screen_size, background_col):
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

        self._shaders = []
        self._current_id = 0

    def addShader(self, vertex_shader, fragment_shader, setup_callback):
        """Construct a shader program in an shader array"""
        self._shaders.append(
            ShaderProgram(vertex_shader, fragment_shader, setup_callback)
        )

    def useShader(self, i: int):
        """Use shader at certain index"""
        self._shaders[i].use()

    def getShaderId(self, i: int):
        """Use shader (identified by the order they are stored)"""
        return self._shaders[i].id

    def __call__(self, i: int):
        """Set shader id"""
        self._current_id = i
        return self

    def __enter__(self):
        """Allows calling with engine(1): to set shader for context."""
        self.useShader(self._current_id)
        return self.getShaderId(self._current_id)

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        pass

    def destroy(self):
        for shader in self._shaders:
            shader.destroy()
