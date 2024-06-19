from typing import Callable

from OpenGL.GL import glEnable, glBlendFunc, glUniform1i, glGetUniformLocation
from OpenGL.GL import GL_BLEND, GL_DEPTH_TEST, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
import numpy as np

from Objects.Balloon import Balloon
from App.GameLoop import GameLoop
from Common.EulerMotion import EulerMotion
from Common.Player import Player
from Common.DynamicOverlay import DynamicOverlay
from Common.PositionCamera import PositionCamera
from Common.ReflectiveLight import ReflectiveLight
from ComputerVision.ModelThread import ModelThread
from Helpers.Globals import (MATERIAL_DEFAULT_GLOBAL_DICT, LIGHT_DEFAULT_GLOBAL_DICT, SCREEN_SIZE,
                             NR_BALLOONS, PITCH_RANGE, ROLL_RANGE, ROLL_RANGE,
                             START_X_RANGE, START_Y, START_Z, MIN_START_DELAY, MAX_START_DELAY,
                             POSSIBLE_HUE_OFFSETS, BALLOON_SCALE, BALLOON_TERMINAL_VELOCITY, BALLOON_DENSITY,
                             FIELD_OF_VIEW, ASPECT_RATIO, CAM_MIN_DISTANCE, CAM_MAX_DISTANCE, CAM_POSITION,
                             LIGHT_POSITION, LIGHT_COLOUR, LIGHT_STRENGTH, LIGHT_AMBIENT_STRENGTH,
                             LIGHT_MIN_DISTANCE, LIGHT_MAX_DISTANCE)


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


def setup_app(obj_file: str, update_callback: Callable):
    capture = ModelThread(0)

    def shape_factory():
        random_x = np.random.random() * START_X_RANGE - START_X_RANGE / 2
        random_pitch = np.random.random() * PITCH_RANGE - PITCH_RANGE / 2
        random_roll = np.random.random() * ROLL_RANGE - ROLL_RANGE / 2
    
        motion = EulerMotion(
            [random_x, START_Y, START_Z],
            [random_pitch, 0, random_roll],
            object_id="motion"
        )

        spawn_time = np.random.randint(MIN_START_DELAY, MAX_START_DELAY)

        hue_offset = np.random.choice(POSSIBLE_HUE_OFFSETS)
        color_variation = {
            'ObjFiles\\Balloon\\string.png': 0,
            'ObjFiles\\Balloon\\rubber.png': hue_offset
        }

        return Balloon(
            obj_file, motion, BALLOON_SCALE, color_variation, BALLOON_TERMINAL_VELOCITY,
            BALLOON_DENSITY, spawn_time, **MATERIAL_DEFAULT_GLOBAL_DICT,
        )

    shape_shader_args = (
        'Shaders/motion.vert',
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
        FIELD_OF_VIEW, ASPECT_RATIO, CAM_MIN_DISTANCE, CAM_MAX_DISTANCE,
        CAM_POSITION, object_id="projection", position_glob_id="cameraPosition"
    )
    player = Player(camera, object_id="view")
    light = ReflectiveLight(LIGHT_POSITION, LIGHT_COLOUR, LIGHT_STRENGTH, LIGHT_AMBIENT_STRENGTH,
                            LIGHT_MIN_DISTANCE, LIGHT_MAX_DISTANCE, **LIGHT_DEFAULT_GLOBAL_DICT)

    app = GameLoop(shape_factory, shape_shader_args,
                   overlay_factory, overlay_shader_args,
                   player, light, capture,
                   limit_frame_rate=True, main_loop_command=update_callback,
                   screen_size=SCREEN_SIZE, nr_balloons=NR_BALLOONS, draw3d=True)
    return app
