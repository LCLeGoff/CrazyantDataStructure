import pandas as pd

from ExperimentGroupBuilder import ExperimentGroupBuilder
from Geometry import Geometry as Geo
geo = Geo()


class ComputeTrajectory:
	def __init__(self, root, group):
		self.experiment = ExperimentGroupBuilder(root).build(group)

	def centered_x_y(self, id_exp_list=None):
		if id_exp_list is None:
			id_exp_list = self.experiment.id_exp_list
		self.experiment.load(['x0', 'y0', 'entrance', 'food_center', 'mm2px'])
		x = self.experiment.x0.series - self.experiment.food_center.array['x']
		y = self.experiment.y0.series - self.experiment.food_center.array['y']
		x /= self.experiment.mm2px.array['mm2px']
		y /= self.experiment.mm2px.array['mm2px']

		for id_exp in id_exp_list:
			entrance_pts1 = [self.experiment.entrance.array.loc[id_exp, 'x1'], self.experiment.entrance.array.loc[id_exp, 'y1']]
			entrance_pts2 = [self.experiment.entrance.array.loc[id_exp, 'x2'], self.experiment.entrance.array.loc[id_exp, 'y2']]
			u = Geo.pts2vect(entrance_pts1, entrance_pts2)
			v = Geo.pts2vect(
				[self.experiment.food_center.array.loc[id_exp, 'x'], self.experiment.food_center.array.loc[id_exp, 'y']],
				Geo.segment_center(entrance_pts1, entrance_pts2))
			a = geo.angle(u, v)
			print(a)
