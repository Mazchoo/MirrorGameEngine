
import pygame as pg
from OpenGL.GL import (glClear, glDisable, glEnable)
from OpenGL.GL import (GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
                       GL_CULL_FACE)
import numpy as np

from Helpers.KeyUtil import get_directional_key_combination
from Common.MultiShaderGameEngine import MultiShaderGameEngine


class GameLoop:

    def __init__(self, shape_factory, shape_args, overlay_factory, overlay_args, player, light,
                 capture, background_col=(.1, .2, .2, 1.), screen_size=(640, 480),
                 limit_frame_rate=True, main_loop_command=lambda x: x):

        self.engine = MultiShaderGameEngine(screen_size, background_col)
        self.engine.addShader(*shape_args)
        self.engine.addShader(*overlay_args)

        self.capture = capture

        with self.engine(0) as shape_shader_id:
            self.light = light
            self.shape = shape_factory()
            self.player = player

            self.light.bind_global_variable_names(shape_shader_id)
            self.shape.bind_global_variable_names(shape_shader_id)
            self.player.bind_global_variable_names(shape_shader_id)

        with self.engine(1):
            self.overlay = overlay_factory()

        with self.engine(0):
            self.player.set_view_to_global()
            self.player.camera.set_projection_to_global()
            self.light.set_all_globals()

        self.limit_frame_rate = limit_frame_rate
        self.last_time = pg.time.get_ticks()
        self.num_frames = 0
        self.frame_time = 1.

        self.screen_width, self.screen_height = screen_size
        self.center_screen = (self.screen_width // 2, self.screen_height // 2)

        self.capture.start()
        self.main_loop(main_loop_command)

    def handle_keys(self):
        direction_modifier = get_directional_key_combination(pg.key.get_pressed())

        if direction_modifier is not None:
            delta_postion = [
                self.frame_time * 0.0025 * np.sin(self.player.theta + direction_modifier),
                0,
                self.frame_time * 0.0025 * np.cos(self.player.theta + direction_modifier)
            ]

            self.player.increment_position(*delta_postion)
            self.player.recalculate_player_view(position=True)
            self.player.set_view_to_global()

    def main_loop(self, loop_callback):
        running = True
        while running:
            if [event for event in pg.event.get() if event.type == pg.QUIT]:
                running = False

            loop_callback(self)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glDisable(GL_DEPTH_TEST)
            glDisable(GL_CULL_FACE)
            self.engine.useShader(1)
            self.overlay.draw()
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_CULL_FACE)

            self.engine.useShader(0)

            self.handle_keys()
            self.shape.draw()

            pg.display.flip()

            self.calculate_frame_rate()
            if self.limit_frame_rate:
                self.engine.clock.tick(60)

        self.quit()

    def calculate_frame_rate(self):
        self.current_time = pg.time.get_ticks()
        delta = self.current_time - self.last_time

        if delta >= 1000:
            frame_rate = max(1, int(1000. * self.num_frames / delta))
            pg.display.set_caption(f'Running at {frame_rate} fps')
            self.last_time = self.current_time
            self.frame_time = 1000. / max(1., frame_rate)
            self.num_frames = 0

        self.num_frames += 1

    def quit(self):
        self.shape.destroy()
        self.engine.destroy()
        self.overlay.destroy()
        self.capture.stop()
        pg.quit()