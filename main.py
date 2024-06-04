
from OpenGL.GL import glEnable, glBlendFunc, glUniform1i, glGetUniformLocation
from OpenGL.GL import GL_BLEND, GL_DEPTH_TEST, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
import cv2
import numpy as np

from Objects.Balloon import Balloon
from App.GameLoop import GameLoop
from Common.EulerMotion import EulerMotion
from Common.Player import Player
from Common.DynamicOverlay import DynamicOverlay
from Common.PositionCamera import PositionCamera
from Common.ReflectiveLight import ReflectiveLight
from ComputerVision.ModelThread import ModelThread
from Helpers.Globals import (MATERIAL_DEFAULT_GLOBAL_DICT, LIGHT_DEFAULT_GLOBAL_DICT,
                             SCREEN_SIZE, RELEASE_MODE)

'''
    TODO - Support Drawing multiple objects
    TODO - Add randomisation to the factory producing objects
    TODO - Make balloons bounce off each other
    TODO - Add tilting to balloons
    TODO - Add compression shader to some objects
    TODO - Add spawn schedule to objects
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
    app.engine.useShader(1)
    app.shape.update()
    app.shape.update_bbox_and_centroid(app.player)
    frame = app.capture.frame

    if not RELEASE_MODE:
        cog = app.shape.screen_centroid
        bbox = app.shape.screen_bbox

        frame = frame.copy() # Use local copy of frame
        cv2.polylines(frame, [bbox], True, (0, 0, 255))
        cv2.circle(frame, cog, radius=2, color=(255, 0, 0), thickness=1)

    app.shape.check_collision(app.capture.pose_dict)

    app.overlay.setTexture(frame)


def main(mesh_name: str):

    capture = ModelThread(0)

    motion_model = EulerMotion([1, 2, -4], [0, 0, 0], object_id="motion")
    shape_factory = lambda: Balloon(
        mesh_name, motion_model, 0.5, 0.075, 1., **MATERIAL_DEFAULT_GLOBAL_DICT,
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

    camera = PositionCamera(
        fovy=45, aspect=640 / 480, near=.1, far=10, position=(0, 0, 0),
        object_id="projection", position_glob_id="cameraPosition"
    )
    player = Player(camera, object_id="view")
    light = ReflectiveLight([0, 2, -3], [1, 1, 1], 2., 1.0, 1.0, 8.0,
                            **LIGHT_DEFAULT_GLOBAL_DICT)

    app = GameLoop(shape_factory, shape_shader_args,
                   overlay_factory, overlay_shader_args, player, light, 
                   capture, limit_frame_rate=True,
                   main_loop_command=update, screen_size=SCREEN_SIZE, draw3d=True)
    return app


if __name__ == '__main__':
    main('ObjFiles/Balloon/balloon.obj')
