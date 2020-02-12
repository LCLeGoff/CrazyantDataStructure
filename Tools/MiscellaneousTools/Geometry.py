import numpy as np
import pandas as pd
import cmath


def pts2vect(p, q):
    return np.array(q) - np.array(p)


def segment_center(p, q):
    return (np.array(p) + np.array(q)) / 2.


def get_line_df(pts1, pts2):
    if pts1.x == pts2.x:
        if pts1.y == pts2.y:
            raise ValueError('pts1 and pts2 are identical')
        else:
            return np.nan, pts1.x
    else:
        if pts1.y == pts2.y:
            return 0, pts1.y
        else:
            a = (pts1.y-pts2.y)/(pts1.x-pts2.x)
            b = pts1.y-a*pts1.x
            return a, b


def get_line(pts1, pts2):
    if pts1[0] == pts2[0]:
        if pts1[1] == pts2[1]:
            raise ValueError('pts1 and pts2 are identical')
        else:
            return np.nan, pts1[0]
    else:
        if pts1[1] == pts2[1]:
            return 0, pts1[1]
        else:
            a = (pts1[1]-pts2[1])/(pts1[0]-pts2[0])
            b = pts1[1]-a*pts1[0]
            return a, b


def get_line_tab(pts1, pts2):
    a = (pts1[:, 1] - pts2[:, 1]) / (pts1[:, 0] - pts2[:, 0])
    b = pts1[:, 1] - a * pts1[:, 0]

    mask = np.where(pts1[:, 1] == pts2[:, 1])[0]
    a[mask] = 0
    b[mask] = pts1[mask, 1]

    mask = np.where(pts1[:, 0] == pts2[:, 0])[0]
    a[mask] = np.nan
    b[mask] = pts1[mask, 1]

    mask = np.where((pts1[:, 0] == pts2[:, 0])*(pts1[:, 1] == pts2[:, 1]))[0]
    a[mask] = np.nan
    b[mask] = np.nan

    return a, b


def convert2vect(u):
    if isinstance(u, list) and isinstance(u[0], tuple):
        u2 = np.array(u)
    elif isinstance(u, list) or isinstance(u, tuple) or u.ndim == 1:
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
    if v is None:
        u2 = convert2vect(u)
        return norm_angle_tab(-np.arctan2(-u2[:, 1], u2[:, 0]))
    else:
        return norm_angle_tab(np.arctan2(cross2d(u, v), dot2d(u, v)))


def cross2d_df(u, v):
    u2 = u.copy()
    v2 = v.copy()
    u2.columns = ['x', 'y']
    v2.columns = ['x', 'y']
    return u2.x * v2.y - u2.y * v2.x


def dot2d_df(u, v):
    u2 = u.copy()
    v2 = v.copy()
    u2.columns = ['x', 'y']
    v2.columns = ['x', 'y']
    return u2.x * v2.x + u2.y * v2.y


def angle_df(u, v=None):
    if v is None:
        u2 = u.copy()
        u2.columns = ['x', 'y']
        return -np.arctan2(-u2.y, u2.x)
    else:
        return np.arctan2(cross2d_df(u, v), dot2d_df(u, v))


def squared_distance(p, q=None):
    if q is None:
        return p[:, 0] ** 2 + p[:, 1] ** 2
    else:
        p2 = convert2vect(p)
        q2 = convert2vect(q)
        return (p2[:, 0] - q2[:, 0]) ** 2 + (p2[:, 1] - q2[:, 1]) ** 2


def squared_distance_df(p, q=None):
    p2 = p.copy()
    p2.columns = ['x', 'y']
    if q is None:
        return p2.x ** 2 + p2.y ** 2
    else:
        q2 = q.copy()
        q2.columns = ['x', 'y']
        return (p2.x - q2.x) ** 2 + (p2.y - q2.y) ** 2


def distance(p, q=None):
    return np.sqrt(squared_distance(p, q))


def distance_df(p, q=None):
    return np.sqrt(squared_distance_df(p, q))


def convert_polar2cartesian(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y


def convert_cartesian2polar(x, y):
    xy_array = np.c_[x, y]
    r = distance(xy_array)
    phi = angle(np.array([1, 0]), xy_array)
    return r, phi


# def norm_angle(theta):
#     if theta > np.pi:
#         return theta - 2 * np.pi
#     elif theta < -np.pi:
#         return theta + 2 * np.pi
#     else:
#         return theta


def norm_angle(theta):
    return np.angle(np.exp(1j*theta))


def norm_angle_df(theta):
    theta2 = theta.copy()
    theta2[:] = np.angle(np.exp(1j*theta))
    return theta2


def norm_angle_tab2(theta):
    theta = np.where(theta > np.pi/2., theta-np.pi)
    theta = np.where(theta < -np.pi/2., theta+np.pi)
    return theta


def norm_vect(vect):
    v = convert2vect(vect)
    return v/distance(v)


def norm_vect_df(vect):
    v = vect.copy()
    d = distance_df(vect)
    v[v.columns[0]] /= d
    v[v.columns[1]] /= d
    return v


def is_in_polygon(pts, path):
    return path.contains_point(pts)


def angle_distance(phi, theta):
    return np.angle(np.exp(1j*(phi-theta)))


def angle_sum(phi, theta):
    return np.angle(np.exp(1j*(phi+theta)))


def angle_distance_df(phi, theta, column_name=None):
    if column_name is None:
        column_name = 'angle_distance'

    if isinstance(phi, pd.DataFrame):
        phi2 = phi.copy()[phi.columns[0]]
    else:
        phi2 = phi.copy()

    if isinstance(theta, pd.DataFrame):
        theta2 = theta.copy()[theta.columns[0]]
    else:
        theta2 = theta.copy()

    df_res = np.exp(1j*(phi2-theta2))
    df_res = pd.DataFrame(np.angle(df_res), index=df_res.index, columns=[column_name])
    return df_res


def angle_sum_tab(phis):
    return cmath.phase(sum(cmath.rect(1, phi) for phi in phis))


def angle_mean(phis):
    return cmath.phase(sum(cmath.rect(1, phi) for phi in phis)/len(phis))


def distance_between_point_and_line(p, line):
    p0 = np.array(p)
    p1 = np.array(line[0])
    p2 = np.array(line[1])
    return np.cross(p2-p1, p0-p1)/distance(p2, p1)


def distance_between_point_and_line_df(p, line):
    p0 = p.copy()
    p1 = line[0].copy()
    p2 = line[1].copy()

    p0.columns = ['x', 'y']
    p1.columns = ['x', 'y']
    p2.columns = ['x', 'y']
    return np.abs(cross2d_df(p2-p1, p0-p1))/distance_df(p2, p1)


def is_counter_clockwise(a, b, c):
    return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)


def is_intersecting_df(a, b, c, d):
    a2 = a.copy()
    a2.columns = ['x', 'y']

    b2 = b.copy()
    b2.columns = ['x', 'y']

    c2 = c.copy()
    c2.columns = ['x', 'y']

    d2 = d.copy()
    d2.columns = ['x', 'y']

    return (is_counter_clockwise(a2, c2, d2) != is_counter_clockwise(b2, c2, d2))\
        & (is_counter_clockwise(a2, b2, c2) != is_counter_clockwise(a2, b2, d2))


def rotation(pts, theta, pts0=None):
    if pts0 is None:
        pts0 = [0, 0]
    pts1 = pts-pts0
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    pts1[0] = pts1[0]*cos_theta+pts1[1]*sin_theta
    pts1[1] = -pts1[0]*sin_theta+pts1[1]*cos_theta
    return pts1+pts0


def rotation_tab(pts, theta, pts0=None):
    if pts0 is None:
        pts0 = [0, 0]
    pts1 = pts-pts0
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    pts1[:, 0] = pts1[:, 0]*cos_theta+pts1[:, 1]*sin_theta
    pts1[:, 1] = -pts1[:, 0]*sin_theta+pts1[:, 1]*cos_theta
    return pts1+pts0


def rotation_df(pts, theta, pts0=None):

    if pts0 is None:
        pts0 = pd.Series(data={'x': [0], 'y': [0]})

    pts1 = pts.copy()
    pts1.columns = ['x', 'y']
    pts1 = pts1-pts0

    cos_theta = np.cos(theta[theta.columns[0]])
    sin_theta = np.sin(theta[theta.columns[0]])

    pts2 = pts1.copy()
    pts2.x = pts1.x*cos_theta+pts1.y*sin_theta
    pts2.y = -pts1.x*sin_theta+pts1.y*cos_theta

    pts2.x += pts0.x
    pts2.y += pts0.y
    return pts2+pts0


# def projection_on_line(p, line):
#
#     p = np.array(p)
#
#     a = np.array(line[0])
#     b = np.array(line[1])
#
#     line_vector = b - a
#     line_vector = norm_vect(line_vector)
#
#     bh = line_vector * dot2d(p - b, line_vector)
#
#     proj_vect = b-p - bh
#
#     return proj_vect

