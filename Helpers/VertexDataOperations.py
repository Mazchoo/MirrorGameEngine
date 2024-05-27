
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
