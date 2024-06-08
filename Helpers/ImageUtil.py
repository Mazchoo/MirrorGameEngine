from typing import Union, List

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


def draw_balloon_bounding_boxes(balloons: list, frame: np.ndarray):
    ''' Draw all bounding boxes on a copy of screen frame. '''
    frame = frame.copy() 
    for balloon in balloons:
        cog = balloon.screen_centroid
        bbox = balloon.screen_bbox

        cv2.polylines(frame, [bbox], True, (0, 0, 255))
        cv2.circle(frame, cog, radius=2, color=(255, 0, 0), thickness=1)

    return frame
