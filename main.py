
from OpenGL.GL import glEnable, glBlendFunc, glUniform1i, glGetUniformLocation
from OpenGL.GL import GL_BLEND, GL_DEPTH_TEST, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA

from Common.ObjMtlMesh import ObjMtlMesh
from App.GameLoop import GameLoop
from Common.EulerMotion import EulerMotion
from Common.Player import Player
from Common.DynamicOverlay import DynamicOverlay
from Common.PositionCamera import PositionCamera
from Common.ReflectiveLight import ReflectiveLight
from ComputerVision.ModelThread import ModelThread
from Helpers.Globals import MATERIAL_DEFAULT_GLOBAL_DICT, LIGHT_DEFAULT_GLOBAL_DICT, SCREEN_SIZE

'''
    TODO - Set the light location based on the most probably light location in the image
    TODO - Support Drawing multiple objects
    TODO - Add collision detection with mouse or limbs
    TODO - Add gravity to some objects
    TODO - Add collision momentum to some objects
    TODO - Add explosion shader to some objects
    TODO - Add movement schedule to objects
    TODO - Add ability to scroll background image
    TODO - Let objects scroll with the screen
'''

def setup3DObjectShader(_shader_id):
    glEnable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


def setupOverlayShader(shader_id):
    # Set up some alpha blending
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    # Set a global variable in the shader
    glUniform1i(glGetUniformLocation(shader_id, "imageTexture"), 0)


def update(app):
    app.engine.useShader(0)
    app.player.camera.position = app.player.position
    app.player.camera.set_position_to_global()

    app.engine.useShader(1)
    app.overlay.setTexture(app.capture.frame)


def main(mesh_name: str):

    capture = ModelThread(0)

    shape_factory = lambda: ObjMtlMesh(
        mesh_name, motion_model, **MATERIAL_DEFAULT_GLOBAL_DICT
    )

    shape_shader_args = (
        'Shaders/specular.vert',
        'Shaders/material.frag',
        setup3DObjectShader
    )

    overlay_shader_args = (
        'Shaders/texture.vert',
        'Shaders/texture.frag',
        setupOverlayShader
    )
    overlay_factory = lambda: DynamicOverlay()

    motion_model = EulerMotion([0, -1, -5], [0, 0, 0], object_id="motion")
    camera = PositionCamera(
        fovy=45, aspect=640 / 480, near=.1, far=10, position=(0, 0, 0),
        object_id="projection", position_glob_id="cameraPosition"
    )
    player = Player(camera, object_id="view")
    light = ReflectiveLight([0, 2, -3], [1, 1, 1], 2., 1.0, 1.0, 8.0,
                            **LIGHT_DEFAULT_GLOBAL_DICT)

    app = GameLoop(shape_factory, shape_shader_args,
                   overlay_factory, overlay_shader_args,
                   player, light, 
                   capture, limit_frame_rate=True, main_loop_command=update, screen_size=SCREEN_SIZE)
    return app


if __name__ == '__main__':
    main('ObjFiles/Bulbasaur/Bulbasaur.obj')
