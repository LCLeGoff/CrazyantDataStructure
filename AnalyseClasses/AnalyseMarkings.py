import numpy as np
import pandas as pd

from pandas import IndexSlice as IdxSc

from matplotlib import pylab as plt
from Builders.ExperimentGroupBuilder import ExperimentGroupBuilder


class AnalyseMarkings:
	def __init__(self, root, group):
		self.exp = ExperimentGroupBuilder(root).build(group)

	def compute_xy_marking(self):
		self.exp.load(['x', 'y', 'markings'])
		self.exp.filter(name1='x', name2='markings', new_name='x_markings')
		self.exp.filter(name1='y', name2='markings', new_name='y_markings')
		self.exp.to_2d(
			name1='x_markings', name2='y_markings',
			new_name='xy_markings', new_name1='x', new_name2='y',
			category='Markings', label='marking coordinates', xlabel='x', ylabel='y',
			description='coordinates of ant positions, while marking'
		)
		self.exp.write('xy_markings')

	def compute_xy_marking_polar(self):
		self.exp.load(['r', 'phi', 'markings'])
		self.exp.filter(name1='r', name2='markings', new_name='r_markings')
		self.exp.filter(name1='phi', name2='markings', new_name='phi_markings')
		self.exp.to_2d(
			name1='r_markings', name2='phi_markings',
			new_name='polar_markings', new_name1='r', new_name2='phi',
			category='Markings', label='marking polar coordinates', xlabel='r', ylabel='phi',
			description='polar coordinates of ant positions, while marking'
		)
		self.exp.write('polar_markings')

	def spatial_repartition_xy_markings(self):
		self.exp.load(['xy_markings'])
		self.exp.plot_repartition('xy_markings')

	def spatial_repartition_xy_markings_2d_hist(self):
		self.exp.load(['xy_markings'])
		self.exp.plot_repartition_hist('xy_markings')

	def marking_interval(self):
		self.exp.load(['markings'])
		name = 'marking_interval'
		ant_exp_array = self.exp.markings.get_id_exp_ant_array()
		event = []
		for (id_exp, id_ant) in ant_exp_array:
			marks = np.array(self.exp.markings.get_row(IdxSc[id_exp, id_ant, :]).reset_index())
			lg = len(marks)-1
			event += list(zip(np.full(lg, id_exp), np.full(lg, id_ant), marks[:-1, 2], marks[1:, 2]-marks[:-1, 2]))
		event = pd.DataFrame(event, columns=['id_exp', 'id_ant', 'frame', name])
		event.set_index(['id_exp', 'id_ant', 'frame'], inplace=True)
		self.exp.build1d(
			array=event, name=name, object_type='Events1d', category='Markings',
			label='marking intervals', description='Time intervals between two marking events'
		)
		self.exp.write(name)

	def compute_first_marking_ant(self):
		self.exp.load(['r', 'x', 'y', 'markings', 'xy_markings', 'food_radius', 'mm2px'])
		self.exp.operation('food_radius', 'mm2px', lambda x, y: round(x / y, 2))
		self.exp.copy1d('food_radius', 'radius_min')
		self.exp.radius_min.replace_values(self.exp.food_radius.get_values()+10)
		self.exp.copy1d('food_radius', 'radius_max')
		self.exp.radius_max.replace_values(120)
		self.exp.copy1d('food_radius', 'radius_med')
		self.exp.radius_med.replace_values((self.exp.radius_min.get_values()+self.exp.radius_max.get_values())/2.)

		self.exp.copy1d('r', 'zones')
		self.exp.operation('zones', 'radius_min', lambda x, y: (x - y < 0).astype(int))
		for radius in ['radius_med', 'radius_max']:
			self.exp.copy1d('r', 'zones2')
			self.exp.operation('zones2', radius, lambda x, y: (x - y < 0).astype(int))
			self.exp.operation('zones', 'zones2', lambda x, y: x + y)

		def plot_test_init(self_class, ant, exp, iii):
			fig2, ax2 = plt.subplots()
			ax2.axis('equal')
			theta = np.arange(-np.pi, np.pi + 0.1, 0.1)
			ax2.plot(
				self_class.exp.radius_min.get_value(exp) * np.cos(theta),
				self_class.exp.radius_min.get_value(exp) * np.sin(theta), c='grey')
			ax2.plot(
				self_class.exp.radius_max.get_value(exp) * np.cos(theta),
				self_class.exp.radius_max.get_value(exp) * np.sin(theta), c='grey')
			ax2.plot(
				self_class.exp.radius_med.get_value(exp) * np.cos(theta),
				self_class.exp.radius_med.get_value(exp) * np.sin(theta), c='grey')
			s = zone_event_ant[mask[iii], 2]
			ax2.plot(
				self_class.exp.x.array.loc[exp, ant, :],
				self_class.exp.y.array.loc[exp, ant, :],
				c='grey')
			ax2.plot(
				self_class.exp.x.array.loc[exp, ant, s],
				self_class.exp.y.array.loc[exp, ant, s],
				'x', c='r')
			ax2.plot(
				self_class.exp.xy_markings.array.loc[IdxSc[exp, ant, :], 'x'],
				self_class.exp.xy_markings.array.loc[IdxSc[exp, ant, :], 'y'],
				'o', c='k')
			ax2.plot(
				self_class.exp.xy_markings.array.loc[IdxSc[exp, ant, :s], 'x'],
				self_class.exp.xy_markings.array.loc[IdxSc[exp, ant, :s], 'y'],
				'x', c='y')
			ax2.set_title((exp, ant, mask[iii]))
			return fig2, ax2

		def plot_suite(ax2, self_class, ant, exp, iii, tt0, tt1, tt, c, ls):
			ax2.plot(
				self_class.exp.xy_markings.array.loc[IdxSc[exp, ant, tt0:tt1], 'x'],
				self_class.exp.xy_markings.array.loc[IdxSc[exp, ant, tt0:tt1], 'y'],
				ls, c=c)
			ax2.set_title((exp, ant, mask[iii], tt))

		def plot_suite2(ax2, self_class, ant, exp, tt0, tt, c, ls):
			ax2.plot(
				self_class.exp.xy_markings.array.loc[IdxSc[exp, ant, tt0:], 'x'],
				self_class.exp.xy_markings.array.loc[IdxSc[exp, ant, tt0:], 'y'],
				ls, c=c)
			ax2.set_title((exp, ant, mask[-1], tt))

		ant_exp_dict = self.exp.markings.get_id_exp_ant_dict()
		ant_exp_array = self.exp.markings.get_id_exp_ant_array()
		self.exp.extract_event(name='zones', new_name='zone_event')
		exp_ant_label = []
		# for id_exp in self.exp.id_exp_list:
		for id_exp in [35, 42, 46, 51, 56]:
			t_min = np.inf
			id_fma = None
			for id_ant in ant_exp_dict[id_exp]:
				if (id_exp, id_ant) in ant_exp_array:
					print((id_exp, id_ant))
					zone_event_ant = np.array(self.exp.zone_event.array.loc[id_exp, id_ant, :].reset_index())
					mask = np.where(zone_event_ant[:, -1] == 3)[0]
					for ii in range(len(mask)-1):
						fig, ax = plot_test_init(self, id_ant, id_exp, ii)

						list_zone_temp = list(zone_event_ant[mask[ii]+1:mask[ii+1], -1])
						if 0 in list_zone_temp:
							mark_temp = [0, 0, 0]
							t = zone_event_ant[mask[ii]+1, 2]
							for jj in range(mask[ii]+1, mask[ii+1]):
								t0 = zone_event_ant[jj, 2]
								t1 = zone_event_ant[jj+1, 2]
								mark_temp[zone_event_ant[jj, -1]] += len(self.exp.markings.array.loc[IdxSc[id_exp, id_ant, t0:t1], :])
							t0 = zone_event_ant[mask[ii], 2]
							t1 = zone_event_ant[mask[ii+1], 2]
							if mark_temp[1] > 0 and mark_temp[2] > 0:
								plot_suite(ax, self, id_ant, id_exp, ii, t0, t1, t, 'g', 'o')
								if t_min > t:
									t_min = t
									id_fma = id_ant
							else:
								plot_suite(ax, self, id_ant, id_exp, ii, t0, t1, t, 'r', 'o')

					fig, ax = plot_test_init(self, id_ant, id_exp, -1)
					list_zone_temp = list(zone_event_ant[mask[-1]:, -1])
					if 0 in list_zone_temp:
						mark_temp = [0, 0, 0]
						t = zone_event_ant[mask[-1]+1, 2]
						for jj in range(mask[-1]+1, len(zone_event_ant)-1):
							t0 = zone_event_ant[jj, 2]
							t1 = zone_event_ant[jj+1, 2]
							mark_temp[zone_event_ant[jj, -1]] += len(self.exp.markings.array.loc[IdxSc[id_exp, id_ant, t0:t1], :])
						t0 = zone_event_ant[mask[-1], 2]
						if mark_temp[1] > 0 and mark_temp[2] > 0:
							plot_suite2(ax, self, id_ant, id_exp, t0, t, 'g', 'o')
							if t_min > t:
								t_min = t
								id_fma = id_ant
						else:
							plot_suite2(ax, self, id_ant, id_exp, t0, t, 'r', 'o')
			print('chosen ant:', id_fma, 'time:', t_min)
			plt.show()

			if id_fma is not None:
				exp_ant_label.append((id_exp, id_fma))
		self.exp.copy1d(
			name='markings', new_name='first_markings', category='Markings',
			label='first markings', description='Markings of the first marking ant'
		)
		self.exp.first_markings.array.reset_index(inplace=True)
		self.exp.first_markings.array.set_index(['id_exp', 'id_ant'], inplace=True)
		self.exp.first_markings.array = self.exp.first_markings.array.loc[exp_ant_label, :]
		self.exp.first_markings.array.reset_index(inplace=True)
		self.exp.first_markings.array.set_index(['id_exp', 'id_ant', 'frame'], inplace=True)
		self.exp.first_markings.array.sort_index(inplace=True)

		self.exp.filter(
			name1='x', name2='first_markings', new_name='x_first_markings',
			label='x', category='Markings', description='x coordinates of ant positions, while marking')
		self.exp.filter(
			name1='y', name2='first_markings', new_name='y_first_markings',
			label='y', category='Markings', description='y coordinates of ant positions, while marking')
		self.exp.to_2d(
			name1='x_first_markings', name2='y_first_markings',
			new_name='xy_first_markings', new_name1='x', new_name2='y',
			category='Markings', label='first marking coordinates', xlabel='x', ylabel='y',
			description='coordinates of the first marking ant positions, while marking'
		)

		self.exp.write('first_markings')
		self.exp.write('xy_first_markings')

	def spatial_repartition_first_markings(self):
		self.exp.load(['xy_first_markings'])
		self.exp.plot_repartition('xy_first_markings')

	def spatial_repartition_first_markings_2d_hist(self):
		self.exp.load(['xy_first_markings'])
		self.exp.plot_repartition_hist('xy_first_markings')

	def compute_first_marking_ant_setup_orientation(self, **kwargs):
		self.exp.load(['xy_first_markings', 'setup_orientation'])
		orientations = set(self.exp.setup_orientation.get_values())
		orient_coord = dict()
		orient_coord['S'] = [-200, 0]
		orient_coord['W'] = [0, -140]
		orient_coord['E'] = [0, 140]
		orient_coord['SW'] = [-200, 140]
		for orientation in orientations:
			new_name = 'xy_first_markings_'+orientation
			indexes = np.array(self.exp.setup_orientation.array.loc[lambda df: df.setup_orientation == orientation].index)
			self.exp.copy2d(
				name='xy_first_markings', new_name=new_name,
				new_xname='x', new_yname='y',
				category='Markings',
				label='first markings (setup oriented '+orientation+')', xlabel='x', ylabel='y',
				description='Markings of the first marking ant with the setup oriented '+orientation
			)
			self.exp.__dict__[new_name].array.reset_index().set_index('id_exp', inplace=True)
			self.exp.__dict__[new_name].array = self.exp.__dict__[new_name].array.loc[indexes]
			self.exp.plot_repartition(new_name, **kwargs)
			# self.exp.plot_repartition_hist(new_name)
			plt.plot(orient_coord[orientation][0], orient_coord[orientation][1], 'o', c='r', ms=10)

	def compute_first_marking_ant_setup_orientation_circle(self):
		self.exp.load(['xy_first_markings', 'setup_orientation', 'food_radius'])
		orientations = set(self.exp.setup_orientation.get_values())
		orient_coord = dict()
		orient_coord['S'] = [-200, 0]
		orient_coord['W'] = [0, -140]
		orient_coord['E'] = [0, 140]
		orient_coord['SW'] = [-200, 140]
		for orientation in orientations:
			new_name = 'xy_first_markings_'+orientation
			self.exp.copy2d(
				name='xy_first_markings', new_name=new_name,
				new_xname='x', new_yname='y',
				category='Markings',
				label='first markings (setup oriented '+orientation+')', xlabel='x', ylabel='y',
				description='Markings of the first marking ant with the setup oriented '+orientation
			)
			indexes = np.array(self.exp.setup_orientation.array.loc[lambda df: df.setup_orientation == orientation].index)
			self.exp.__dict__[new_name].array.reset_index().set_index('id_exp', inplace=True)
			self.exp.__dict__[new_name].array = self.exp.__dict__[new_name].array.loc[indexes]

			indexes = np.array(self.exp.xy_first_markings.array.loc[lambda df: df.x**2+df.y**2 < 110**2].index)
			self.exp.__dict__[new_name].array = self.exp.__dict__[new_name].array.loc[indexes]

			self.exp.plot_repartition(new_name)
			# self.exp.plot_repartition_hist(new_name)
			plt.plot(orient_coord[orientation][0], orient_coord[orientation][1], 'o-', c='r', ms=10)
