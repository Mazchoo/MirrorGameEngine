
import numpy as np
import pygame as pg


def get_directional_key_combination(keys):
    """
    Get direction through combination of keys.

    w: 1 -> 0 degrees
    a: 2 -> 90 degrees
    w & a: 3 -> 45 degrees
    s: 4 -> 180 degrees
    w & s: 5 -> x
    a & s: 6 -> 135 degrees
    w & a & s: 7 -> 90 degrees
    d: 8 -> 270 degrees
    w & d: 9 -> 315 degrees
    a & d: 10 -> x
    w & a & d: 11 -> 0 degrees
    s & d: 12 -> 225 degrees
    w & s & d: 13 -> 270 degrees
    a & s & d: 14 -> 180 degrees
    w & a & s & d: 15 -> x
    """

    key_combination = 0
    up = keys[pg.K_UP]
    down = keys[pg.K_DOWN]

    if keys[pg.K_w]:
        key_combination += 1
    if keys[pg.K_a]:
        key_combination += 2
    if keys[pg.K_s]:
        key_combination += 4
    if keys[pg.K_d]:
        key_combination += 8

    if key_combination > 0:
        if key_combination == 1:
            return 0., up, down
        elif key_combination == 3:
            return np.pi * 0.25, up, down
        elif key_combination == 2 or key_combination == 7:
            return np.pi * 0.5, up, down
        elif key_combination == 6:
            return np.pi * 0.75, up, down
        elif key_combination == 4 or key_combination == 14:
            return np.pi, up, down
        elif key_combination == 12:
            return np.pi * 1.25, up, down
        elif key_combination == 8 or key_combination == 13:
            return np.pi * 1.5, up, down
        elif key_combination == 9:
            return np.pi * 1.75, up, down

    return None, up, down
