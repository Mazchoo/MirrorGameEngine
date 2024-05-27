
from OpenGL.GL import glEnable, glBlendFunc, glUniform1i, glGetUniformLocation
from OpenGL.GL import GL_BLEND, GL_DEPTH_TEST, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
import cv2
import numpy as np

from Common.ObjMtlMesh import ObjMtlMesh
from App.GameLoop import GameLoop
from Common.EulerMotion import EulerMotion
from Common.Player import Player
from Common.DynamicOverlay import DynamicOverlay
from Common.PositionCamera import PositionCamera
from Common.ReflectiveLight import ReflectiveLight
from ComputerVision.ModelThread import ModelThread
from Helpers.Globals import MATERIAL_DEFAULT_GLOBAL_DICT, LIGHT_DEFAULT_GLOBAL_DICT, SCREEN_SIZE, IMAGE_SIZE, RELEASE_MODE

'''
    TODO - Support Drawing multiple objects
    TODO - Add collision detection with mouse or limbs
    TODO - Add gravity to some objects
    TODO - Add collision momentum to some objects
    TODO - Add explosion shader to some objects
    TODO - Add movement schedule to objects
    TODO - Set the light location based on the most probably light location in the image
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

IMAGE_SIZE_NP = np.array(IMAGE_SIZE, dtype=np.float32)

def update(app):
    app.engine.useShader(0)
    app.player.camera.position = app.player.position
    app.player.camera.set_position_to_global()

    app.engine.useShader(1)
    frame = app.capture.frame

    if not RELEASE_MODE:
        v_min, v_max = app.shape.bbox
        v_min = v_min.copy()
        v_max = v_max.copy()
        avg_z = (v_min[2] + v_max[2]) / 2
        v_min[2] = avg_z
        v_max[2] = avg_z
        v_min = app.shape.motion.transform_vertex(v_min)
        v_max = app.shape.motion.transform_vertex(v_max)
        v_min = app.player.transform_vertex(v_min)
        v_max = app.player.transform_vertex(v_max)
        v_min = app.player.camera.transform_vertex(v_min)
        v_max = app.player.camera.transform_vertex(v_max)
        v_min /= v_min[2]
        v_max /= v_max[2]
        v_min = v_min[:2]
        v_max = v_max[:2]
        v_min[1] *= -1
        v_max[1] *= -1
        v_min = v_min * 0.5 + 0.5
        v_max = v_max * 0.5 + 0.5
        v_min *= IMAGE_SIZE_NP
        v_max *= IMAGE_SIZE_NP

        bbox = np.vstack([v_min, v_max])
        new_v_min = bbox.min(axis=0) - (1, 1)
        new_v_max = bbox.max(axis=0) + (1, 1)

        v_min_tuple = tuple(new_v_min.astype(np.int32))
        v_max_tuple = tuple(new_v_max.astype(np.int32))
        cv2.rectangle(frame, v_min_tuple, v_max_tuple, (0, 0, 255))

    app.overlay.setTexture(frame)

def main(mesh_name: str):

    capture = ModelThread(0)

    motion_model = EulerMotion([0, 1, -4], [0, 0, 0], object_id="motion")
    shape_factory = lambda: ObjMtlMesh(
        mesh_name, motion_model, **MATERIAL_DEFAULT_GLOBAL_DICT, normalize_scale=0.5
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
                   overlay_factory, overlay_shader_args,
                   player, light, 
                   capture, limit_frame_rate=True, main_loop_command=update, screen_size=SCREEN_SIZE)
    return app


if __name__ == '__main__':
    main('ObjFiles/Balloon/balloon.obj')
