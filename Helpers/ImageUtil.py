from typing import Union

import numpy as np
import pygame as pg
import cv2


def read_image_data_from_source(texture_source: Union[str, tuple], hue_offset=0):
    if hue_offset is None:
        hue_offset = 0
    hue_offset = int(hue_offset)

    if isinstance(texture_source, str):
        texture_array = cv2.imread(texture_source)
        texture_array = cv2.cvtColor(texture_array, cv2.COLOR_BGR2HSV)
        texture_array[:, :, 0] = cv2.add(texture_array[:, :, 0], hue_offset)
        texture_array = cv2.cvtColor(texture_array, cv2.COLOR_HSV2RGB)

    elif isinstance(texture_source, tuple):
        texture_array = np.array([[texture_source]], dtype=np.float32)

    else:
        raise ValueError('Invalid texture source')

    image = pg.surfarray.make_surface(texture_array)
    width, height = image.get_rect().size
    return pg.image.tostring(image, "RGBA"), width, height
