import numpy as np
from Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Tools.Geometry import pts2vect, angle, distance
from math import pi


class AnalyseTrajectory:
	def __init__(self, root, group):
		self.exp = ExperimentGroupBuilder(root).build(group)

	def centered_x_y(self, id_exp_list=None):
		if id_exp_list is None:
			id_exp_list = self.exp.id_exp_list
		self.exp.load(['x0', 'y0', 'entrance1', 'entrance2', 'food_center', 'mm2px', 'traj_translation'])
		self.exp.copy(
			name='x0', new_name='x',
			category='Trajectory',
			label='x',
			description='x coordinate (in the food system)'
		)
		self.exp.copy(
			name='y0', new_name='y',
			category='Trajectory',
			label='y',
			description='y coordinate (in the food system)'
		)

		self.exp.operation('x', 'food_center', lambda x, y: x - y, 'x')
		self.exp.operation('y', 'food_center', lambda x, y: x - y, 'y')
		self.exp.operation('x', 'traj_translation', lambda x, y: x + y, 'x')
		self.exp.operation('y', 'traj_translation', lambda x, y: x + y, 'y')
		self.exp.operation('x', 'mm2px', lambda x, y: round(x / y, 2))
		self.exp.operation('y', 'mm2px', lambda x, y: round(x / y, 2))

		for id_exp in id_exp_list:
			food_center = np.array(self.exp.food_center.get_value(id_exp))
			entrance_pts1 = np.array(self.exp.entrance1.get_value(id_exp))
			entrance_pts2 = np.array(self.exp.entrance1.get_value(id_exp))
			u = pts2vect(entrance_pts1, entrance_pts2)
			a1 = angle([1, 0], u)
			entrance_pts1 -= food_center
			if abs(a1) < pi/4:
				a2 = angle([1, 0], entrance_pts1)
				if abs(a2) > pi/2:
					# fig, ax = plt.subplots()
					# ax.plot(self.exp.x.array.loc[id_exp, :, :]['x'], self.exp.y.array.loc[id_exp, :, :]['y'])
					self.exp.x.operation_on_id_exp(id_exp, lambda z: z*-1)
					self.exp.y.operation_on_id_exp(id_exp, lambda z: z*-1)
					# fig, ax = plt.subplots()
					# ax.plot(self.exp.x.array.loc[id_exp, :, :]['x'], self.exp.y.array.loc[id_exp, :, :]['y'])
					# plt.show()
			else:
				a2 = angle([0, 1], entrance_pts1)
				if abs(a2) > pi/2:
					self.exp.x.operation_on_id_exp(id_exp, lambda z: z*-1)
					self.exp.y.operation_on_id_exp(id_exp, lambda z: z*-1)

		self.exp.write(['x', 'y'])

	def xy_polar(self):
		self.exp.load(['x', 'y'])
		self.exp.to_2d(
			name1='x', name2='y',
			new_name='xy', new_name1='x', new_name2='y',
			category='Trajectory', label='coordinates', xlabel='x', ylabel='y',
			description='coordinates of ant positions'
		)
		self.exp.copy(
			name='x', new_name='r',
			category='Trajectory',
			label='r',
			description='radial coordinate (in the food system)'
		)
		self.exp.copy(
			name='x', new_name='phi',
			category='Trajectory',
			label='phi',
			description='angular coordinate (in the food system)'
		)
		self.exp.r.replace_values(np.around(distance([0, 0], self.exp.xy.get_array()), 3))
		self.exp.phi.replace_values(np.around(angle([1, 0], self.exp.xy.get_array()), 3))

		self.exp.write(['r', 'phi'])
