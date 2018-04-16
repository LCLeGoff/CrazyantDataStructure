import numpy as np
import pandas as pd
import scipy.stats as scs
import scipy.signal as scsig

from pandas import IndexSlice as IdxSc

from matplotlib import pylab as plt
from Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from DataObjects.Events1d import Events1dBuilder
from Plotter.ColorObject import ColorObject
from Plotter.Plotter1d import Plotter1d


class AnalyseMarkings:
	def __init__(self, root, group):
		self.exp = ExperimentGroupBuilder(root).build(group)

	def compute_xy_marking(self):
		print('xy_marking')
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
		print('xy_marking_polar')
		self.exp.load(['r', 'phi', 'markings'])
		self.exp.filter(
			name1='r', name2='markings', new_name='r_markings',
			category='Markings', label='marking radial coordinates',
			description='radial coordinates of ant positions, while marking')
		self.exp.filter(
			name1='phi', name2='markings', new_name='phi_markings',
			category='Markings', label='marking angular coordinates',
			description='angular coordinates of ant positions, while marking')
		self.exp.to_2d(
			name1='r_markings', name2='phi_markings',
			new_name='polar_markings', new_name1='r', new_name2='phi',
			category='Markings', label='marking polar coordinates', xlabel='r', ylabel='phi',
			description='polar coordinates of ant positions, while marking'
		)
		self.exp.write('r_markings')
		self.exp.write('phi_markings')
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
		print(name)
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

	def compute_radial_zones(self):
		self.exp.load(['r', 'food_radius', 'mm2px'])
		self.exp.operation('food_radius', 'mm2px', lambda x, y: round(x / y, 2))
		self.exp.copy1d('food_radius', 'radius_min')
		self.exp.radius_min.replace_values(self.exp.food_radius.get_values()+15)
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

	def plot_radial_zones(self, exp, ant, t):
		fig, ax = plt.subplots()
		ax.axis('equal')
		ax.set_title('exp:'+str(exp)+', ant:'+str(ant)+', t:'+str(t))
		theta = np.arange(-np.pi, np.pi + 0.1, 0.1)
		for radius in ['radius_min', 'radius_med', 'radius_max']:
			ax.plot(
				self.exp.__dict__[radius].get_value(exp) * np.cos(theta),
				self.exp.__dict__[radius].get_value(exp) * np.sin(theta),
				c='grey')
		return fig, ax

	def plot_radial_zones_3panels(self, exp, ant, t):
		fig, ax = plt.subplots(1, 3, figsize=(10, 5))
		fig.subplots_adjust(left=0, right=1)
		for i in range(3):
			ax[i].axis('equal')
			ax[i].set_title('exp:'+str(exp)+', ant:'+str(ant)+', t:'+str(t))
			theta = np.arange(-np.pi, np.pi + 0.1, 0.1)
			for radius in ['radius_min', 'radius_med', 'radius_max']:
				ax[i].plot(
					self.exp.__dict__[radius].get_value(exp) * np.cos(theta),
					self.exp.__dict__[radius].get_value(exp) * np.sin(theta),
					c='grey')
		return fig, ax

	def plot_traj(self, ax, exp, ant, c, t0=None, t1=None, ms=None):
		if ms is None:
			ms = 5
		if t0 is None:
			t0 = 0
		if t1 is None:
			ax.plot(
				self.exp.x.array.loc[exp, ant, t0:],
				self.exp.y.array.loc[exp, ant, t0:],
				c=c, ms=ms)
		else:
			ax.plot(
				self.exp.x.array.loc[exp, ant, t0:t1],
				self.exp.y.array.loc[exp, ant, t0:t1],
				c=c, ms=ms)

	def plot_mark(self, ax, exp, ant, c, t0=None, t1=None):
		if t0 is None:
			t0 = 0
		if t1 is None:
			ax.plot(
				self.exp.xy_markings.array.loc[IdxSc[exp, ant, t0:], 'x'],
				self.exp.xy_markings.array.loc[IdxSc[exp, ant, t0:], 'y'],
				'o', ms=5, c=c)
		else:
			ax.plot(
				self.exp.xy_markings.array.loc[IdxSc[exp, ant, t0:t1], 'x'],
				self.exp.xy_markings.array.loc[IdxSc[exp, ant, t0:t1], 'y'],
				'o', ms=5, c=c)

	def plot_previous_loops_around_food(self, ax, exp, ant, zone_event_ant, mask, iii):
		t = zone_event_ant[mask[iii], 2]
		self.plot_traj(ax, exp, ant, 'k', t1=t)
		self.plot_mark(ax, exp, ant, '0.3', t1=t)

	def plot_next_loops_around_food(self, ax, exp, ant, zone_event_ant, mask, iii):
		if iii < len(mask)-1:
			t = zone_event_ant[mask[iii+1], 2]+1
			self.plot_traj(ax, exp, ant, '0.7', t)
			self.plot_mark(ax, exp, ant, 'wheat', t)

	def plot_current_loop_around_food(self, ax, exp, ant, zone_event_ant, mask, iii, c1, c2):
		if iii < len(mask)-1:
			t0, t1 = zone_event_ant[[mask[iii], mask[iii+1]], 2]
			self.plot_traj(ax, exp, ant, c2, t0, t1)
			self.plot_mark(ax, exp, ant, c1, t0, t1)
		else:
			t = zone_event_ant[mask[iii], 2]
			self.plot_traj(ax, exp, ant, c2, t)
			self.plot_mark(ax, exp, ant, c1, t)

	def plot_interesting_loop(self, exp, ant, zone_event_ant, mask, iii, t, c1, c2):
		fig, ax = self.plot_radial_zones(exp, ant, t)
		self.plot_previous_loops_around_food(ax, exp, ant, zone_event_ant, mask, iii)
		self.plot_next_loops_around_food(ax, exp, ant, zone_event_ant, mask, iii)
		self.plot_current_loop_around_food(ax, exp, ant, zone_event_ant, mask, iii, c1, c2)

	def compute_zone_visit(self, id_exp, id_ant, zone_event, mask, ii):
		if ii == len(mask)-1:
			list_ii = range(mask[-1]+1, len(zone_event)-1)
		else:
			list_ii = range(mask[ii]+1, mask[ii+1])
		zone_visit = [0, 0, 0]
		for jj in list_ii:
			t0 = zone_event[jj, 2]
			t1 = zone_event[jj + 1, 2]
			zone_visit[zone_event[jj, -1]] += len(self.exp.markings.array.loc[IdxSc[id_exp, id_ant, t0:t1], :])
		return zone_visit

	def radial_criterion(self, id_exp, id_ant, zone_event, mask, ii, t_min, id_fma):
		t = zone_event[mask[ii]+1, 2]
		zone_visit = self.compute_zone_visit(id_exp, id_ant, zone_event, mask, ii)
		if zone_visit[1] > 0 and zone_visit[2] > 0:
			# self.plot_interesting_loop(id_exp, id_ant, zone_event, mask, ii, t, 'g', 'y')
			if t_min > t:
				t_min = t
				id_fma = id_ant
		# else:
			# self.plot_interesting_loop(id_exp, id_ant, zone_event, mask, ii, t, 'r', 'orange')
		return t_min, id_fma

	def compute_first_marking_ant_radial_criterion(self, id_exp_list=None, show=False):
		if id_exp_list is None:
			id_exp_list = self.exp.id_exp_list

		self.exp.load(['x', 'y', 'markings', 'xy_markings'])
		self.compute_radial_zones()

		ant_exp_dict = self.exp.markings.get_id_exp_ant_dict()
		ant_exp_array = self.exp.markings.get_id_exp_ant_array()
		self.exp.extract_event(name='zones', new_name='zone_event')
		exp_ant_label = []
		for id_exp in id_exp_list:
			t_min = np.inf
			id_fma = None
			for id_ant in ant_exp_dict[id_exp]:
				if (id_exp, id_ant) in ant_exp_array:
					print((id_exp, id_ant))
					zone_event_ant = np.array(self.exp.zone_event.array.loc[id_exp, id_ant, :].reset_index())
					mask = np.where(zone_event_ant[:, -1] == 3)[0]
					if len(mask) != 0:
						for ii in range(len(mask)-1):
							list_zone_temp = list(zone_event_ant[mask[ii]+1:mask[ii+1], -1])
							if 0 in list_zone_temp:
								t_min, id_fma = self.radial_criterion(id_exp, id_ant, zone_event_ant, mask, ii, t_min, id_fma)

						list_zone_temp = list(zone_event_ant[mask[-1]:, -1])
						if 0 in list_zone_temp:
							t_min, id_ant = self.radial_criterion(id_exp, id_ant, zone_event_ant, mask, len(mask)-1, t_min, id_fma)
			print('chosen ant:', id_fma, 'time:', t_min)
			if show:
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

	def compute_batch_threshold(self, id_exp, id_ant):
		self.exp.load('marking_interval')
		mark_interval_ant = np.array(self.exp.marking_interval.array.loc[id_exp, id_ant, :].reset_index())
		# n_occ, times = np.histogram(mark_interval_ant[:, -1], bins=range(0, 1000, 10))
		# plt.loglog((times[1:]+times[:-1])/2., n_occ/np.sum(n_occ), '.-')
		n_occ, times = np.histogram(mark_interval_ant[:, -1], bins='fd')
		# plt.loglog((times[1:]+times[:-1])/2., n_occ/np.sum(n_occ), '.-', c='k')
		# plt.show()
		times2 = np.arange(1, 1000)

		mask0 = np.where(n_occ == 0)[0]
		if len(mask0) == 0:
			n_occ = np.array(list(n_occ)+[0])
			times = np.array(list(times)+[times[-1]+1])
			mask0 = np.where(n_occ == 0)[0]
		thresh0 = times[mask0[0]]
		if thresh0 == 0 and len(mask0) > 1:
			thresh0 = times[mask0[1]]

		# print(1./np.sum(n_occ))
		# kde = scs.gaussian_kde(np.array(self.exp.marking_interval.get_row_id_exp_ant(id_exp, id_ant)).T[0], 1./np.sum(n_occ)*2)
		# y_kde = kde.pdf(times2)
		# idx = scsig.argrelextrema(y_kde, np.less)[0]
		# thresh2 = times2[idx[1]]

		mask = np.where(n_occ > 1)[0]
		if len(mask) == 0:
			thresh1 = times[mask0[0]]
			if thresh1 == 0 and len(mask0) > 1:
				thresh1 = times[mask0[1]]
		else:
			times3 = times[mask[-1]:]
			n_occ2 = n_occ[mask[-1]:]
			mask0 = np.where(n_occ2 == 0)[0]
			if len(mask0) == 0:
				thresh1 = times3[-1]
			else:
				thresh1 = times3[np.where(n_occ2 == 0)[0][0]]
		thresh1 = min(thresh1, 200)

		# fig, ax = plt.subplots()
		# ax.loglog((times[1:]+times[:-1])/2., n_occ/np.sum(n_occ), '.-', c='k')
		# # plt.loglog(times2, y_kde)
		# # plt.loglog(times2[idx], y_kde[idx], 'o')
		# ax.axvline(thresh0, ls='-', c='grey')
		# ax.axvline(thresh1, ls=':', c='grey')
		# ax.axvline(200, ls='--', c='grey')
		# ax.set_title('exp:'+str(id_exp)+', ant:'+str(id_ant))
		# # ax.set_ylim((1e-4, 1))
		# # ax.axvline(200, c='grey')
		# # return 200
		return thresh0, thresh1, 200

	def compute_first_marking_ant_batch_criterion(self, id_exp_list=None, show=False):
		if id_exp_list is None:
			id_exp_list = self.exp.id_exp_list

		self.exp.load(['x', 'y', 'markings', 'xy_markings', 'r_markings'])
		self.compute_radial_zones()

		ant_exp_dict = self.exp.markings.get_id_exp_ant_dict()
		ant_exp_array = self.exp.markings.get_id_exp_ant_array()
		self.exp.extract_event(name='zones', new_name='zone_event')

		# for id_exp in [3, 4, 6, 9, 10, 11, 26, 27, 42, 30, 33, 35, 36, 42, 46, 48, 49, 51, 52, 53, 56, 58]:
		# for id_exp in [3]:
		thresh_hist = []
		for id_exp in id_exp_list:
			id_mfa = [None, None, None]
			batches_mfa = [[], [], []]
			t_min = [np.inf, np.inf, np.inf]
			for id_ant in ant_exp_dict[id_exp]:
				if (id_exp, id_ant) in ant_exp_array:
					print((id_exp, id_ant))
					thresh_list = self.compute_batch_threshold(id_exp, id_ant)
					thresh_hist.append(thresh_list[1])
					zone_event_ant = np.array(self.exp.zone_event.array.loc[id_exp, id_ant, :].reset_index())
					mask = np.where(zone_event_ant[:, -1] == 3)[0]

					for ii in range(len(mask)-1):
						list_zone_temp = list(zone_event_ant[mask[ii]+1:mask[ii+1], -1])
						if 0 in list_zone_temp:
							t0, t1 = zone_event_ant[[mask[ii], mask[ii+1]], 2]
							id_mfa, batches_mfa, t_min = self.batch_criterion(
								id_exp, id_ant, id_mfa, batches_mfa, t_min, thresh_list, zone_event_ant, mask, ii, t0, t1)

					t0 = zone_event_ant[mask[-1], 2]
					id_mfa, batches_mfa, t_min = self.batch_criterion(
						id_exp, id_ant, id_mfa, batches_mfa, t_min, thresh_list, zone_event_ant, mask, len(mask)-1, t0)
			print('ant chosen:'+str(id_mfa), t_min)
			# self.plot_chosen_batch(id_exp, id_mfa, t_min, batches_mfa)
			if show:
				plt.show()
		y, x = np.histogram(thresh_hist, bins=range(0, 201, 2))
		plt.plot(x[1:], y)
		plt.show()


	def plot_batches(self, id_exp, id_ant, batches_list, zone_event_ant, mask, ii, t0=None, t1=None):
		fig, ax = self.plot_radial_zones_3panels(id_exp, id_ant, t0)
		for i in range(3):
			self.plot_previous_loops_around_food(ax[i], id_exp, id_ant, zone_event_ant, mask, ii)
			self.plot_next_loops_around_food(ax[i], id_exp, id_ant, zone_event_ant, mask, ii)

			self.plot_traj(ax[i], id_exp, id_ant, 'y', t0, t1)
			cols = ColorObject.create_cmap('jet', len(batches_list[i]))
			for i_col, batches in enumerate(batches_list[i]):
				self.plot_mark(ax[i], id_exp, id_ant, cols[i_col], batches[0][2], batches[-1][2])
			for line in ax[i].lines:
				x_data, y_data = line.get_xdata(), line.get_ydata()
				line.set_xdata(y_data)
				line.set_ydata(x_data)
			ax[i].invert_xaxis()

	def plot_chosen_batch(self, id_exp, id_ant, t, batches_mfa):
		fig, ax = self.plot_radial_zones_3panels(id_exp, id_ant, t)
		for i in range(3):
			batch2plot = batches_mfa[i]
			if len(batch2plot) != 0:
				self.plot_mark(ax[i], id_exp, id_ant, 'g', batch2plot[0][2], batch2plot[-1][2])
			for line in ax[i].lines:
				x_data, y_data = line.get_xdata(), line.get_ydata()
				line.set_xdata(y_data)
				line.set_ydata(x_data)
			ax[i].invert_xaxis()

	def batch_criterion(self, id_exp, id_ant, id_mfa, batches_mfa, t_min, thresh_list, zone_event, mask, ii, t0, t1=None):
		min_lg_rad = 70

		xy_mark = np.array(self.exp.xy_markings.get_row_id_exp_ant_in_frame_interval(id_exp, id_ant, t0, t1).reset_index())
		if len(xy_mark) != 0:
			batches_list = []
			for i, thresh in enumerate(thresh_list):
				batches = [[list(xy_mark[0, :])]]
				for jj in range(1, len(xy_mark)):
					if xy_mark[jj, 2] - xy_mark[jj-1, 2] < thresh:
						batches[-1].append(list(xy_mark[jj, :]))
					else:
						batches += [[list(xy_mark[jj, :])]]
				jj = -1
				rad_min = 0
				rad_max = 0
				while jj < len(batches)-1 and rad_max-rad_min < min_lg_rad:
					jj = jj + 1
					batch = batches[jj]
					if len(batch) > 3:
						rad_min = self.exp.r.get_row_id_exp_ent_frame_from_array(np.array(batch)[:, :3]).min()['r']
						rad_max = self.exp.r.get_row_id_exp_ent_frame_from_array(np.array(batch)[:, :3]).max()['r']

				if rad_max-rad_min >= min_lg_rad:
					batch = batches[jj]
					zone_mark = \
						self.exp.r_markings.get_row_id_exp_ant_frame(id_exp, id_ant, batch[0][2])\
						- self.exp.radius_max.get_value(id_exp)
					if int(zone_mark) < 0:
						t = batch[0][2]
						if t < t_min[i]:
							t_min[i] = t
							batches_mfa[i] = batch
							id_mfa[i] = id_ant

				batches_list.append(batches)
			# self.plot_batches(id_exp, id_ant, batches_list, zone_event, mask, ii, t0, t1)
		return id_mfa, batches_mfa, t_min

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
