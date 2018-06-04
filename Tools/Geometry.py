import numpy as np


def pts2vect(p, q):
    return np.array(q) - np.array(p)


def segment_center(p, q):
    return (np.array(p) + np.array(q)) / 2.


def convert2vect(u):
    if isinstance(u, list) or u.ndim == 1:
        u2 = np.array(u)[np.newaxis]
    else:
        u2 = u.copy()
    return u2


def cross2d(u, v):
    u2 = convert2vect(u)
    v2 = convert2vect(v)
    return u2[:, 0] * v2[:, 1] - u2[:, 1] * v2[:, 0]


def dot2d(u, v):
    u2 = convert2vect(u)
    v2 = convert2vect(v)
    return u2[:, 0] * v2[:, 0] + u2[:, 1] * v2[:, 1]


def angle(u, v):
    return np.arctan2(cross2d(u, v), dot2d(u, v))


def squared_distance(p, q):
    p2 = convert2vect(p)
    q2 = convert2vect(q)
    return (p2[:, 0] - q2[:, 0]) ** 2 + (p2[:, 1] - q2[:, 1]) ** 2


def distance(p, q):
    return np.sqrt(squared_distance(p, q))
