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


def angle(u, v=None):
    return np.arctan2(cross2d(u, v), dot2d(u, v))


def squared_distance(p, q=None):
    if q is None:
        return p[:, 0] ** 2 + p[:, 1] ** 2
    else:
        p2 = convert2vect(p)
        q2 = convert2vect(q)
        return (p2[:, 0] - q2[:, 0]) ** 2 + (p2[:, 1] - q2[:, 1]) ** 2


def distance(p, q=None):
    return np.sqrt(squared_distance(p, q))


def convert_polar2cartesian(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y


def convert_cartesian2polar(x, y):
    xy_array = np.c_[x, y]
    r = distance(xy_array)
    phi = angle(np.array([1, 0]), xy_array)
    return r, phi


def norm_angle(theta):
    if theta > np.pi:
        return theta - 2 * np.pi
    elif theta < -np.pi:
        return theta + 2 * np.pi
    else:
        return theta
