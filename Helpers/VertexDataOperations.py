
import numpy as np


def normalize_l1(vertices: np.ndarray, factor=2):
    max_norm = np.abs(vertices).max()
    max_norm /= factor
    vertices /= max_norm


def normalize_l2(vertices: np.ndarray, factor=2):
    max_norm = np.sqrt(np.square(vertices).max())
    max_norm /= factor
    vertices /= max_norm
