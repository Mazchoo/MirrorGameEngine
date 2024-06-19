
import numpy as np


def normalize_l1(vertices: np.ndarray, factor=2):
    ''' Scale all vertices down by maximum value. '''
    max_norm = np.abs(vertices).max()
    max_norm /= factor
    vertices /= max_norm


def get_bbox_2d(vertices: np.ndarray):
    ''' Get bounding box of vertices (min and max homogenous 4d coodinate) '''
    v_mins = np.concatenate([vertices[:, :3].min(axis=0), [1]])
    v_maxs = np.concatenate([vertices[:, :3].max(axis=0), [1]])
    bbox = np.vstack([v_mins, v_maxs])
    avg_z = bbox[:, 2].mean()
    bbox[:, 2] = avg_z
    return bbox


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


def convex_volume(vertices: np.ndarray):
    ''' Assumes centroid is at (0, 0, 0) '''
    if len(vertices) < 3 or len(vertices) % 3 != 0:
        raise ValueError("Calculating centroid of array without faces")

    a = vertices[::3]
    b = vertices[1::3]
    c = vertices[2::3]
    concat_vertices = np.hstack([a, b, c])
    concat_vertices = concat_vertices.reshape(len(vertices) // 3, 3, 3)
    determinants = np.linalg.det(concat_vertices)
    return determinants.sum() / 6

