
import pygame as pg
from OpenGL.GL import (glClear, glUseProgram, glDeleteProgram, glEnable, glBlendFunc)
from OpenGL.GL import (GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                       GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
import numpy as np

from Helpers.KeyUtil import get_directional_key_combination
from Common.SingleShaderEngine import InspectionEngine


class InspectionApp:

    def __init__(self, shape_factory, vertex_shader, fragment_shader, player, light,
                 background_col=(.1, .2, .2, 1.), screen_size=(640, 480), limit_frame_rate=True,
                 main_loop_command=lambda x: x):

        self.engine = InspectionEngine(screen_size, background_col, vertex_shader, fragment_shader)

        self.light = light
        self.shape = shape_factory()
        self.player = player

        glUseProgram(self.engine.shader)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.light.bind_global_variable_names(self.engine.shader)
        self.shape.bind_global_variable_names(self.engine.shader)
        self.player.bind_global_variable_names(self.engine.shader)

        self.player.set_view_to_global()
        self.player.camera.set_projection_to_global()
        self.light.set_all_globals()

        self.limit_frame_rate = limit_frame_rate
        self.last_time = pg.time.get_ticks()
        self.frame_freq = 0
        self.frame_time = 1.

        self.screen_width, self.screen_height = screen_size
        self.center_screen = (self.screen_width // 2, self.screen_height // 2)

        self.main_loop(main_loop_command)

    def handle_keys(self):
        direction_modifier, up, down = get_directional_key_combination(pg.key.get_pressed())
        dir_movement = direction_modifier is not None

        if dir_movement or up or down:
            delta_x = np.sin(self.player.theta + direction_modifier) if dir_movement else 0.
            delta_z = np.cos(self.player.theta + direction_modifier) if dir_movement else 0.

            if up:
                delta_y = -1.
            elif down:
                delta_y = 1.
            else:
                delta_y = 0.
        
            delta_postion = [
                self.frame_time * 0.0025 * delta_x,
                self.frame_time * 0.0025 * delta_y,
                self.frame_time * 0.0025 * delta_z
            ]

            self.player.increment_position(*delta_postion)
            self.player.recalculate_player_view(position=True)
            self.player.set_view_to_global()

    def handle_mouse(self):

        (x, y) = pg.mouse.get_pos()
        theta_increment = self.frame_time * 0.00020 * (self.center_screen[0] - x)
        phi_increment = self.frame_time * 0.00015 * (self.center_screen[1] - y)

        self.player.increment_angles(theta_increment, phi_increment)
        self.player.recalculate_player_view(theta=True, phi=True)
        self.player.set_view_to_global()

        pg.mouse.set_pos(self.center_screen)

    def main_loop(self, loop_callback):
        running = True
        glUseProgram(self.engine.shader)

        while running:
            if [event for event in pg.event.get() if event.type == pg.QUIT]:
                running = False

            self.handle_keys()
            self.handle_mouse()

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.shape.draw()
            loop_callback(self)

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

    def quit(self):
        self.shape.destroy()
        glDeleteProgram(self.engine.shader)
        pg.quit()
