
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
    TODO - Make balloons bounce off each other
    TODO - Add compression shader to some objects
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
    for s in app.shapes:
        s.update()
        s.update_bbox_and_centroid(app.player)
    frame = app.capture.frame

    if not RELEASE_MODE and frame is not None:
        for s in app.shapes:
            cog = s.screen_centroid
            bbox = s.screen_bbox

            frame = frame.copy() # Use local copy of frame
            cv2.polylines(frame, [bbox], True, (0, 0, 255))
            cv2.circle(frame, cog, radius=2, color=(255, 0, 0), thickness=1)

    for s in app.shapes:
        s.check_collision(app.capture.pose_dict)

    if frame is not None:
        app.overlay.setTexture(frame)


def main(mesh_name: str):

    capture = ModelThread(0)

    def shape_factory():
        random_x = np.random.random() * 2 - 1
        random_pitch = np.random.random() * 0.1 - 0.05
        random_roll = np.random.random() * np.pi * 2
        motion = EulerMotion([random_x, 2.25, -4], [random_pitch, 0, random_roll], object_id="motion")
        spawn_time = np.random.randint(120, 400)
        hue_offset = np.random.choice([12, 20, 0, 0, 0, 60, 25])
        color_variation = {
            'ObjFiles\\Balloon\\string.png': 0,
            'ObjFiles\\Balloon\\rubber.png': hue_offset
        }
        balloon = Balloon(
            mesh_name, motion, 0.5, color_variation,
            0.075, 1., spawn_time, **MATERIAL_DEFAULT_GLOBAL_DICT,
        )
        return balloon

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
                   capture, limit_frame_rate=True, main_loop_command=update,
                   screen_size=SCREEN_SIZE, nr_shapes=3, draw3d=True)
    return app


if __name__ == '__main__':
    main('ObjFiles/Balloon/balloon.obj')
