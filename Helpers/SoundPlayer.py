from os import listdir
from pathlib import Path
import pygame as pg


class SoundPlayer:

    def __init__(self):
        self.sound_files = {}
        for path in listdir('./Sounds'):
            self.sound_files[path] = pg.mixer.Sound(f'./Sounds/{path}')

    def play_sound(self, name: str):
        pg.mixer.Sound.play(self.sound_files[name])


if __name__ == '__main__':
    pg.init()
    player = SoundPlayer()
    player.play_sound('balloon-pop.wav')
