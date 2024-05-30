
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
                             SCREEN_SIZE, IMAGE_SIZE, RELEASE_MODE)

'''
    TODO - Add despawn criteria
    TODO - Support Drawing multiple objects
    TODO - Add randomisation to the factory producing objects
    TODO - Add collision detection with mouse or limbs
    TODO - Add collision momentum to some objects
    TODO - Make balloons bounce off walls
    TODO - Make balloons bounce off each other
    TODO - Add tilting to balloons
    TODO - Add compression shader to some objects
    TODO - Add spawn schedule to objects
    TODO - Set the light location based on the most probably light location in the image
    TODO - The player view matrix is not needed
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

IMAGE_SIZE_NP = np.array(IMAGE_SIZE, dtype=np.float32)

def update(app):
    app.engine.useShader(0)
    app.shape.update()

    app.engine.useShader(1)
    frame = app.capture.frame

    if not RELEASE_MODE:
        v_min, v_max = app.shape.bbox
        v_min = app.transform_vertex_to_screen(v_min)
        v_max = app.transform_vertex_to_screen(v_max)
        cog = app.transform_vertex_to_screen(np.array([0, 0, 0, 1], dtype=np.float32))

        bbox = np.vstack([v_min, v_max])
        new_v_min = bbox.min(axis=0) - (1, 1)
        new_v_max = bbox.max(axis=0) + (1, 1)
        vertex_list = np.vstack([
            [new_v_min, [new_v_min[0], new_v_max[1]], new_v_max, [new_v_max[0], new_v_min[1]]]
        ])

        rot = app.shape.motion.angles[1]
        rot_mat = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]], dtype=np.float32)
        vertex_list = (vertex_list - cog) @ rot_mat + cog

        frame = frame.copy() # Use local copy of frame
        cv2.polylines(frame, [vertex_list.astype(np.int32)], True, (0, 0, 255))
        cv2.circle(frame, cog.astype(np.int32), radius=2, color=(255, 0, 0), thickness=1)

    app.overlay.setTexture(frame)

def main(mesh_name: str):

    capture = ModelThread(0)

    motion_model = EulerMotion([1, 1, -4], [0, 0, 0], object_id="motion")
    shape_factory = lambda: Balloon(
        mesh_name, motion_model, 0.5, 2., 1., **MATERIAL_DEFAULT_GLOBAL_DICT,
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
    light = ReflectiveLight([0, 0, -3], [1, 1, 1], 2., 1.0, 1.0, 8.0,
                            **LIGHT_DEFAULT_GLOBAL_DICT)

    app = GameLoop(shape_factory, shape_shader_args,
                   overlay_factory, overlay_shader_args, player, light, 
                   capture, limit_frame_rate=True,
                   main_loop_command=update, screen_size=SCREEN_SIZE, draw3d=True)
    return app


if __name__ == '__main__':
    main('ObjFiles/Balloon/balloon.obj')
