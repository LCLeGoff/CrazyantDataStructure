import numpy as np

from matplotlib import pylab as plt
from sklearn.neighbors import NearestNeighbors
from pandas import IndexSlice as IdxSc
from Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from PandasIndexManager.PandasIndexManager import PandasIndexManager
from Plotter.ColorObject import ColorObject


class Recruitment:
	def __init__(self, root, group):
		self.pd_idx_manager = PandasIndexManager()
		self.exp = ExperimentGroupBuilder(root).build(group)
		self.arena_radius = 120

	def __compute_xy_radial_zones(self):
		self.exp.load(['r', 'food_radius', 'mm2px'])

		self.__convert_food_radius2mm()
		self.__compute_food_neighborhood_radius()

		self.__compute_is_xy_in_food_neighborhood()
		self.__compute_is_xy_in_circular_arena()

		self.__compute_sum_of_is_xy_in_food_neighborhood_and_in_circular_arena()

	def __compute_sum_of_is_xy_in_food_neighborhood_and_in_circular_arena(self):
		self.exp.add_copy1d('is_xy_in_food_neighborhood', 'xy_radial_zones')
		fct_sum = lambda x, y: x + y
		self.exp.operation_between_2names('xy_radial_zones', 'is_in_circular_arena', fct_sum)

	def __compute_is_xy_in_circular_arena(self):
		self.exp.add_copy1d('r', 'is_in_circular_arena')
		fct_is_lesser_than_arena_radius = lambda x: (x - self.arena_radius < 0).astype(int)
		self.exp.operation('is_in_circular_arena', fct_is_lesser_than_arena_radius)

	def __compute_is_xy_in_food_neighborhood(self):
		self.exp.add_copy1d(name_to_copy='r', copy_name='is_xy_in_food_neighborhood')
		fct_is_lesser_than_circular_arena_radius = lambda x, y: (x - y < 0).astype(int)
		self.exp.operation_between_2names(
			'is_xy_in_food_neighborhood', 'food_neighborhood_radius', fct_is_lesser_than_circular_arena_radius)

	def __compute_food_neighborhood_radius(self):
		self.exp.add_copy1d('food_radius', 'food_neighborhood_radius')
		food_neighborhood_radius_array = self.exp.food_radius.get_values() + 15
		self.exp.food_neighborhood_radius.replace_values(food_neighborhood_radius_array)

	def __convert_food_radius2mm(self):
		fct_convert_in_mm = lambda x, y: round(x / y, 2)
		self.exp.operation_between_2names('food_radius', 'mm2px', fct_convert_in_mm)

	def compute_recruitment(self, id_exp_list=None):
		id_exp_list = self.exp.set_id_exp_list(id_exp_list)

		self.exp.load(['markings', 'xy_markings', 'r_markings'])

		self.__compute_xy_radial_zones()

		dict_of_idx_exp_ant = self.exp.markings.get_id_exp_ant_dict()
		array_of_idx_exp_ant = self.exp.markings.get_id_exp_ant_array()

		self.exp.add_from_event_extraction(name='xy_radial_zones', new_name='zone_change_events')

		self.exp.add_new1d_empty(
			name='recruitment_interval',
			object_type='Events1d', category='Markings',
			label='Recruitment intervals', description='Time intervals when ants are considering recruiting'
		)
		for i, id_exp in enumerate(id_exp_list):
			if id_exp in dict_of_idx_exp_ant:
				print(i, id_exp)
				batches_recruitment = []
				for id_ant in dict_of_idx_exp_ant[id_exp]:
					if (id_exp, id_ant) in array_of_idx_exp_ant:

						thresh = self.__compute_batch_threshold(id_exp, id_ant)
						ant_zone_change_event_array = self.__get_zone_change_events_for_id_ant(id_exp, id_ant)
						mask_where_ant_enter_food_neighborhood = np.where(ant_zone_change_event_array[:, -1] == 2)[0]
						nbr_entry_in_food_neighborhood = len(mask_where_ant_enter_food_neighborhood)
						for ii in range(nbr_entry_in_food_neighborhood-1):
							idx_exit = mask_where_ant_enter_food_neighborhood[ii] + 1
							idx_next_entry = mask_where_ant_enter_food_neighborhood[ii + 1]
							list_visited_zones = list(ant_zone_change_event_array[idx_exit:idx_next_entry, -1])
							is_ant_exit_circular_arena = (0 in list_visited_zones)
							if is_ant_exit_circular_arena:
								t_exit = ant_zone_change_event_array[idx_exit, 2]
								t_next_entry = ant_zone_change_event_array[idx_next_entry, 2]
								batches_recruitment = self.batch_criterion(
									id_exp, id_ant, batches_recruitment, thresh, t_exit, t_next_entry)

						t_last_exit = ant_zone_change_event_array[mask_where_ant_enter_food_neighborhood[-1], 2]
						batches_recruitment = self.batch_criterion(
							id_exp, id_ant, batches_recruitment, thresh, t_last_exit)
				for ii, batch in enumerate(batches_recruitment):
					id_ant = int(batch[0, 1])
					t0 = int(batch[0, 2]) - 1
					dt = int(batch[-1, 2]) - t0 + 2
					self.exp.recruitment_interval.add_row((id_exp, id_ant, t0), dt)
					# self.__plot_recruitment_batch(id_exp, batches_recruitment)
					# plt.show()
		self.exp.write('recruitment_interval')

	def __plot_recruitment_batch(self, id_exp, batches_recruitment):
		fig, ax = self.__plot_radial_zones(id_exp, '', '')
		t_min = np.inf
		batch = []
		cols = ColorObject.create_cmap('jet', len(batches_recruitment))
		for i, batch2plot in enumerate(batches_recruitment):
			id_ant = batch2plot[0][1]
			if batch2plot[0][2] < t_min:
				batch = batch2plot
				t_min = batch2plot[0][2]
			self.__plot_mark(ax, id_exp, id_ant, cols[i], batch2plot[0][2], batch2plot[-1][2])
		if len(batch) != 0:
			self.__plot_mark(ax, id_exp, batch[0][1], 'y', batch[0][2], batch[-1][2])
		for line in ax.lines:
			x_data, y_data = line.get_xdata(), line.get_ydata()
			line.set_xdata(y_data)
			line.set_ydata(x_data)
		ax.invert_xaxis()

	def __plot_mark(self, ax, id_exp, id_ant, c, t0=None, t1=None):
		if t0 is None:
			t0 = 0
		if t1 is None:
			ax.plot(
				self.exp.xy_markings.array.loc[IdxSc[id_exp, id_ant, t0:], 'x'],
				self.exp.xy_markings.array.loc[IdxSc[id_exp, id_ant, t0:], 'y'],
				'o', ms=5, c=c)
		else:
			ax.plot(
				self.exp.xy_markings.array.loc[IdxSc[id_exp, id_ant, t0:t1], 'x'],
				self.exp.xy_markings.array.loc[IdxSc[id_exp, id_ant, t0:t1], 'y'],
				'o', ms=5, c=c)

	def __plot_radial_zones(self, exp, ant, t):
		fig, ax = plt.subplots()
		ax.axis('equal')
		ax.set_title('exp:' + str(exp) + ', ant:' + str(ant) + ', t:' + str(t))
		theta = np.arange(-np.pi, np.pi + 0.1, 0.1)
		for radius in [self.arena_radius, self.exp.food_neighborhood_radius.get_value(exp)]:
			ax.plot(radius * np.cos(theta), radius * np.sin(theta), c='grey')
		return fig, ax

	def __get_zone_change_events_for_id_ant(self, id_exp, id_ant):
		ant_zone_change_event_df = self.exp.zone_change_events.get_row_id_exp_ant(id_exp, id_ant)
		ant_zone_change_event_array = np.array(ant_zone_change_event_df.reset_index())
		return ant_zone_change_event_array

	def __compute_batch_threshold(self, id_exp, id_ant):
		self.exp.load('marking_interval')
		mark_interval_ant = np.array(self.exp.marking_interval.array.loc[id_exp, id_ant, :].reset_index())
		n_occ, times = np.histogram(mark_interval_ant[:, -1], bins='fd')

		mask0 = np.where(n_occ == 0)[0]
		mask1 = np.where(n_occ > 1)[0]
		if len(mask0) == 0:
			thresh1 = 200
		elif len(mask1) == 0:
			thresh1 = np.floor(times[mask0[0]])
			if thresh1 == 0 and len(mask0) > 1:
				thresh1 = np.floor(times[mask0[1]])
		else:
			times3 = times[mask1[-1]:]
			n_occ2 = n_occ[mask1[-1]:]
			mask0 = np.where(n_occ2 == 0)[0]
			if len(mask0) == 0:
				thresh1 = np.floor(times3[-1])
			else:
				thresh1 = np.floor(times3[np.where(n_occ2 == 0)[0][0]])
		thresh1 = max(min(thresh1, 200), 60)

		return thresh1

	def batch_criterion(self, id_exp, id_ant, batches_recruitment, thresh, t0, t1=None):
		min_lg_rad = 60
		max_lg_rad = 70

		xy_mark = np.array(
			self.exp.xy_markings.get_row_id_exp_ant_in_frame_interval(id_exp, id_ant, t0, t1).reset_index())
		if len(xy_mark) != 0:
			batches = [[list(xy_mark[0, :])]]
			for jj in range(1, len(xy_mark)):
				if xy_mark[jj, 2] - xy_mark[jj - 1, 2] < thresh:
					batches[-1].append(list(xy_mark[jj, :]))
				else:
					batches += [[list(xy_mark[jj, :])]]
			batches2 = []
			for batch in batches:
				batch = np.array(batch)
				if len(batch) > 3:
					nn_obj = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(batch[:, -2:])
					nn_dist, nn_indices = nn_obj.kneighbors(batch[:, -2:])
					while len(batch) > 3 and sum(nn_dist[:, 1] > max_lg_rad) != 0:
						mask2 = np.where(nn_dist[:, 1] <= max_lg_rad)[0]
						batch = batch[mask2, :]
						nn_obj = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(batch[:, -2:])
						nn_dist, nn_indices = nn_obj.kneighbors(batch[:, -2:])
					if len(batch) > 3:
						batches2.append(batch.tolist())

			rad_min = 0
			rad_max = 0
			for jj in range(len(batches2)):
				batch = np.array(batches2[jj])
				if len(batch) > 3:
					rad_min = self.exp.r.get_row_id_exp_ant_frame_from_array(batch[:, :3]).min()['r']
					rad_max = self.exp.r.get_row_id_exp_ant_frame_from_array(batch[:, :3]).max()['r']

				if rad_max - rad_min >= min_lg_rad:
					t = batch[0, 2]
					zone_mark = \
						self.exp.r_markings.get_row_id_exp_ant_frame(id_exp, id_ant, t) - self.arena_radius
					if int(zone_mark) < 0:
						r_mark = np.array(
							self.exp.r_markings.get_row_id_exp_ant_in_frame_interval(id_exp, id_ant, t, batch[-1, 2]))
						r_mark = np.sort(r_mark, axis=0)
						r_mark = np.array(r_mark[1:] - r_mark[:-1], dtype=int)
						r_mark = r_mark > max_lg_rad
						if np.sum(r_mark) == 0:
							batches_recruitment.append(batch)
		return batches_recruitment
