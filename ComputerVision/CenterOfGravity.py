
from time import perf_counter

import cv2
import numpy as np

from ComputerVision.CameraThread import CameraThread

# ToDo - Try adding a lot of filtering to this


def draw_center_of_light(image: np.ndarray):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey_image = image.astype(np.float32)
    grey_image = np.clip(grey_image, 127, 255) - 127
    total_grey = grey_image.sum()
    if total_grey > 0:
        grey_image /= total_grey

    height, width = grey_image.shape
    x_mesh, y_mesh = np.meshgrid(np.arange(
        width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    x = round((x_mesh * grey_image).sum())
    y = round((y_mesh * grey_image).sum())
    cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    return image


def draw_largest_blob(image: np.ndarray):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightest_col = np.percentile(grey_image, 75)
    _, threshold = cv2.threshold(grey_image, brightest_col, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest_contour = max(contours, key=cv2.contourArea)
    ((x, y),) = biggest_contour.mean(axis=0)
    cv2.drawContours(image, [biggest_contour], -1, (0, 0, 255), 3)
    cv2.circle(image, (round(x), round(y)), 2, (255, 0, 256), -1)

    return image



if __name__ == '__main__':
    capture = CameraThread(0)
    capture.start()
    cv2.namedWindow("Center of Light")

    total_time = 0
    total_measurements = 0

    while True:
        start = perf_counter()
        pred_frame = draw_largest_blob(capture.frame)
        total_time += perf_counter() - start
        total_measurements += 1

        if total_measurements != 0:
            avg_time_ms = np.round(1000 * total_time / total_measurements, 3)
            avg_fps = 1000 / avg_time_ms
            print(f'Avg time {total_time/total_measurements} fps {avg_fps}')

        cv2.imshow("Center of Light", pred_frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            break