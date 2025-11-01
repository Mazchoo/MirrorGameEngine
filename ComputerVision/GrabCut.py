from time import perf_counter

import cv2
import numpy as np

from ComputerVision.ModelThread import ModelThread

DRAW_BODY_PARTS_KPT_IDS = (
    (2, 1),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
    (16, 17),
    (14, 15),
    (8, 11),
)
kpt_names = [
    "nose",
    "neck",
    "r_sho",
    "r_elb",
    "r_wri",
    "l_sho",
    "l_elb",
    "l_wri",
    "r_hip",
    "r_knee",
    "r_ank",
    "l_hip",
    "l_knee",
    "l_ank",
    "r_eye",
    "l_eye",
    "r_ear",
    "l_ear",
]


def draw_ellipses_on_image(image: np.ndarray, key_points, color, thickness):
    for kp_ind_1, kp_ind_2 in DRAW_BODY_PARTS_KPT_IDS:
        if key_points[kp_ind_1][0] > 0 and key_points[kp_ind_2][0] > 0:
            mid_point = tuple((key_points[kp_ind_1] + key_points[kp_ind_2]) // 2)
            diff_arr = key_points[kp_ind_1] - key_points[kp_ind_2]
            radius = int(np.linalg.norm(diff_arr)) // 2
            angle = 180 - np.degrees(np.arctan2(*diff_arr))

            # Points on the face capture the head
            if kp_ind_1 in {14, 15, 16, 17}:
                short_axis = int(radius * 1.75)
            else:
                short_axis = radius // 2

            cv2.ellipse(
                image, mid_point, (short_axis, radius), angle, 0, 360, color, thickness
            )


def draw_body_regions(frame: np.ndarray, poses: list):
    if not poses:
        return frame

    key_points = poses[0].keypoints
    draw_ellipses_on_image(frame, key_points, (0, 0, 255), 2)
    return frame


def draw_body_mask(frame: np.ndarray, poses: list):
    if not poses:
        return frame

    height, width, _ = frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    key_points = poses[0].keypoints
    draw_ellipses_on_image(mask, key_points, 1, -1)
    frame = frame * mask[:, :, np.newaxis]
    return frame


def draw_mask_grabcut(frame: np.ndarray, poses: list):
    if not poses:
        return frame

    height, width, _ = frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    key_points = poses[0].keypoints
    draw_ellipses_on_image(mask, key_points, 1, -1)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    mask, bgdModel, fgdModel = cv2.grabCut(
        frame, mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK
    )
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    frame = frame * mask[:, :, np.newaxis]

    return frame


def draw_mask_grabcut_rect(frame: np.ndarray, poses: list):
    if not poses:
        return frame

    height, width, _ = frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    key_points = np.array([kp for kp in poses[0].keypoints if kp[0] > 0])
    min_x, min_y = key_points.min(axis=0)
    max_x, max_y = key_points.max(axis=0)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (min_x, min_y - 100, max_x, max_y)
    cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    frame = frame * mask[:, :, np.newaxis]

    return frame


if __name__ == "__main__":
    capture = ModelThread(0)
    capture.start()
    cv2.namedWindow("GrabCut")

    total_time = 0
    total_measurements = 0

    while True:
        start = perf_counter()
        pred_frame = draw_body_mask(capture.frame, capture.model.poses)
        total_time += perf_counter() - start
        total_measurements += 1

        if total_measurements != 0:
            avg_time_ms = np.round(1000 * total_time / total_measurements, 3)
            avg_fps = 1000 / avg_time_ms
            print(f"Avg time {total_time / total_measurements} fps {avg_fps}")

        cv2.imshow("GrabCut", pred_frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            break
