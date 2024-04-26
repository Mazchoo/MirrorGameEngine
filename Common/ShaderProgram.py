
from typing import Callable

from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GL import glUseProgram, glDeleteProgram
from OpenGL.GL import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER


class ShaderProgram:

    def __init__(self, vertex_file_path: str, fragment_file_path: str, setup_callback: Callable):
        self.id = ShaderProgram.createShader(vertex_file_path, fragment_file_path)
        glUseProgram(self.id)
        setup_callback(self.id)

    def use(self):
        glUseProgram(self.id)

    def destroy(self):
        glDeleteProgram(self.id)

    @staticmethod
    def readShaderFile(path: str):
        with open(path, 'r') as f:
            source = f.readlines()
        return source

    @staticmethod
    def createShader(vertex_file_path, fragment_file_path):
        vertex_source = ShaderProgram.readShaderFile(vertex_file_path)
        fragment_source = ShaderProgram.readShaderFile(fragment_file_path)

        return compileProgram(
            compileShader(vertex_source, GL_VERTEX_SHADER),
            compileShader(fragment_source, GL_FRAGMENT_SHADER)
        )
