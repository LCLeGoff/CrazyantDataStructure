import numpy as np

from Geometry import pts2vect, angle
from ExperimentGroupBuilder import ExperimentGroupBuilder
from math import pi


class ComputeTrajectory:
	def __init__(self, root, group):
		self.experiment = ExperimentGroupBuilder(root).build(group)

	def centered_x_y(self, id_exp_list=None):
		if id_exp_list is None:
			id_exp_list = self.experiment.id_exp_list
		self.experiment.load(['x0', 'y0', 'entrance1', 'entrance2', 'food_center', 'mm2px'])
		self.experiment.x0.array['x0'] -= self.experiment.food_center.array['x']

			# .operation(lambda z: z-self.experiment.food_center.array['x'])
		self.experiment.y0.operation(lambda z: z-self.experiment.food_center.array['y'])

		# self.experiment.x0.operation(lambda z: z-self.experiment.mm2px.array['mm2px'])
		# self.experiment.y0.operation(lambda z: z-self.experiment.mm2px.array['mm2px'])
		# for id_exp in id_exp_list:
		# 	food_center = np.array(
		# 		[self.experiment.food_center.array.loc[id_exp, 'x'], self.experiment.food_center.array.loc[id_exp, 'y']])
		# 	entrance_pts1 = np.array(
		# 		[self.experiment.entrance.array.loc[id_exp, 'x1'], self.experiment.entrance.array.loc[id_exp, 'y1']])
		# 	entrance_pts2 = np.array(
		# 		[self.experiment.entrance.array.loc[id_exp, 'x2'], self.experiment.entrance.array.loc[id_exp, 'y2']])
		# 	u = pts2vect(entrance_pts1, entrance_pts2)
		# 	a1 = angle([1, 0], u)
		# 	entrance_pts1 -= food_center
		# 	if abs(abs(a1)-0.5*pi) < abs(a1):
		# 		a2 = angle([1, 0], entrance_pts1)
		# 		if abs(abs(a2)-pi) < abs(a2):
		# 			self.experiment.x0.series.loc[id_exp, :, :] *= -1
		# 			self.experiment.y0.series.loc[id_exp, :, :] *= -1
		# 	else:
		# 		a2 = angle([0, 1], entrance_pts1)
		# 		if abs(abs(a2)-pi) < abs(a2):
		# 			self.experiment.x0.series.loc[id_exp, :, :] *= -1
		# 			self.experiment.y0.series.loc[id_exp, :, :] *= -1
