import numpy as np


class Geometry:
	def __init__(self):
		pass

	@staticmethod
	def pts2vect(p, q):
		return np.array(q)-np.array(p)

	@staticmethod
	def segment_center(p, q):
		return (np.array(p)+np.array(q))/2.

	@staticmethod
	def cross2d(u, v):
		return u[0]*v[1]-u[1]*v[0]

	@staticmethod
	def dot2d(u, v):
		return u[0]*v[0]+u[1]*v[1]

	def angle(self, u, v):
		return np.arctan2(self.cross2d(u, v), self.dot2d(u, v))
