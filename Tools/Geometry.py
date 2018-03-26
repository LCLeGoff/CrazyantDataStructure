import numpy as np


def pts2vect(p, q):
	return np.array(q)-np.array(p)


def segment_center(p, q):
	return (np.array(p)+np.array(q))/2.


def cross2d(u, v):
	return u[0]*v[1]-u[1]*v[0]


def dot2d(u, v):
	return u[0]*v[0]+u[1]*v[1]


def angle(u, v):
	return np.arctan2(cross2d(u, v), dot2d(u, v))
