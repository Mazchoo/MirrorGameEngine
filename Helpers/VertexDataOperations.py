
import numpy as np


def normalize_l1(vertices: np.ndarray, factor=2):
    max_norm = np.abs(vertices).max()
    max_norm /= factor
    vertices /= max_norm


def normalize_l2(vertices: np.ndarray, factor=2):
    max_norm = np.sqrt(np.square(vertices).max())
    max_norm /= factor
    vertices /= max_norm


def get_bbox(vertices: np.ndarray):
    v_mins = np.concatenate([vertices[:, :3].min(axis=0), [1]])
    v_maxs = np.concatenate([vertices[:, :3].max(axis=0), [1]])
    return np.vstack([v_mins, v_maxs])


def centroid_weighted_by_face(vertices: np.ndarray):
    if len(vertices) < 3 or len(vertices) % 3 != 0:
        raise ValueError("Calculating centroid of array without faces")

    a = vertices[::3]
    b = vertices[1::3]
    c = vertices[2::3]
    avg_vertex = (a + b + c) / 3
    ab = a - b
    ac = a - c
    area = np.linalg.norm(np.cross(ab, ac), axis=1)
    return (avg_vertex * np.expand_dims(area, -1)).sum(axis=0) / area.sum()