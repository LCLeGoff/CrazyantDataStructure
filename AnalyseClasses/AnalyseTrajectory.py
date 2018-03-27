import numpy as np

from Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Tools.Geometry import pts2vect, angle
from math import pi


class AnalyseTrajectory:
	def __init__(self, root, group):
		self.exp = ExperimentGroupBuilder(root).build(group)

	def centered_x_y(self, id_exp_list=None):
		if id_exp_list is None:
			id_exp_list = self.exp.id_exp_list
		self.exp.load(['x0', 'y0', 'entrance1', 'entrance2', 'food_center', 'mm2px'])
		self.exp.copy(
			name='x0', new_name='x',
			category='Trajectory',
			object_type=self.exp.x0.object_type,
			label='x',
			description='x coordinate (in the food system)'
		)
		self.exp.copy(
			name='y0', new_name='y',
			category='Trajectory',
			object_type=self.exp.y0.object_type,
			label='y',
			description='y coordinate (in the food system)'
		)

		self.exp.operation('x', 'food_center', lambda x, y: x - y, 'x')
		self.exp.operation('x', 'mm2px', lambda x, y: round(x / y, 3))
		self.exp.operation('y', 'food_center', lambda x, y: x - y, 'y')
		self.exp.operation('y', 'mm2px', lambda x, y: round(x / y, 3))

		for id_exp in id_exp_list:
			food_center = np.array(self.exp.food_center.get_value(id_exp))
			entrance_pts1 = np.array(self.exp.entrance1.get_value(id_exp))
			entrance_pts2 = np.array(self.exp.entrance1.get_value(id_exp))
			u = pts2vect(entrance_pts1, entrance_pts2)
			a1 = angle([1, 0], u)
			entrance_pts1 -= food_center
			if abs(abs(a1)-0.5*pi) < abs(a1):
				a2 = angle([1, 0], entrance_pts1)
				if abs(abs(a2)-pi) < abs(a2):
					self.exp.x.operation_on_id_exp(id_exp, lambda z: z*-1)
					self.exp.y.operation_on_id_exp(id_exp, lambda z: z*-1)
			else:
				a2 = angle([0, 1], entrance_pts1)
				if abs(abs(a2)-pi) < abs(a2):
					self.exp.x.operation_on_id_exp(id_exp, lambda z: z*-1)
					self.exp.y.operation_on_id_exp(id_exp, lambda z: z*-1)

		self.exp.write(['x', 'y'])
