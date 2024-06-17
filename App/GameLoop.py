from typing import Callable

import pygame as pg
from OpenGL.GL import (glClear, glDisable, glEnable)
from OpenGL.GL import (GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
                       GL_CULL_FACE)

from Common.OverlayEngine import OverlayEngine
from Helpers.SoundPlayer import SoundPlayer


def create_all_balloons(shape_factory: Callable, n: int) -> list:
    balloons = []
    total_spawn_time = 0
    for _ in range(n):
        balloon = shape_factory()
        total_spawn_time += balloon.spawn_time
        balloon.spawn_count = total_spawn_time
        balloons.append(balloon)
    return balloons

class GameLoop:

    def __init__(self, shape_factory, shape_args, overlay_factory, overlay_args, player, light,
                 capture, background_col=(.1, .2, .2, 1.), screen_size=(640, 480), nr_balloons=10,
                 limit_frame_rate=True, main_loop_command=lambda x: x, draw3d=True):

        self.engine = OverlayEngine(screen_size, background_col)
        self.engine.addShader(*shape_args)
        self.engine.addShader(*overlay_args)

        self.capture = capture

        self.sound_player = SoundPlayer()

        with self.engine(0) as shape_shader_id:
            self.light = light
            self.balloons = create_all_balloons(shape_factory, nr_balloons)
            self.player = player

            self.light.bind_global_variable_names(shape_shader_id)
            for balloon in self.balloons:
                balloon.bind_global_variable_names(shape_shader_id) 
            self.player.bind_global_variable_names(shape_shader_id)

        with self.engine(1):
            self.overlay = overlay_factory()

        with self.engine(0):
            self.player.set_view_to_global()
            self.player.camera.set_projection_to_global()
            self.light.set_all_globals()

        self.limit_frame_rate = limit_frame_rate
        self.last_time = pg.time.get_ticks()
        self.total_frames = 0
        self.frame_freq = 0
        self.frame_time = 1.

        self.screen_width, self.screen_height = screen_size
        self.center_screen = (self.screen_width // 2, self.screen_height // 2)
        self.draw3d = draw3d

        self.capture.start()
        self.main_loop(main_loop_command)

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

            if self.draw3d:
                for balloon in self.balloons:
                    balloon.draw()

            pg.display.flip()

            self.calculate_frame_rate()
            if self.limit_frame_rate:
                self.engine.clock.tick(60)

        self.quit()

    def calculate_frame_rate(self):
        self.current_time = pg.time.get_ticks()
        delta = self.current_time - self.last_time

        if delta >= 1000:
            frame_rate = max(1, int(1000. * self.frame_freq / delta))
            pg.display.set_caption(f'Running at {frame_rate} fps')
            self.last_time = self.current_time
            self.frame_time = 1000. / max(1., frame_rate)
            self.frame_freq = 0

        self.frame_freq += 1
        self.total_frames += 1

    def quit(self):
        for balloon in self.balloons:
            balloon.destroy()
        self.engine.destroy()
        self.overlay.destroy()
        self.capture.stop()
        pg.quit()
