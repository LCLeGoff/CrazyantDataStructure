import numpy as np

from matplotlib import pylab as plt
from sklearn.neighbors import NearestNeighbors
from pandas import IndexSlice as IdxSc
from Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from PandasIndexManager.PandasIndexManager import PandasIndexManager
from Plotter.ColorObject import ColorObject
from Tools.Geometry import distance


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
		food_neighborhood_radius_df = self.exp.food_radius.get_values() + 15
		self.exp.food_neighborhood_radius.replace_values(food_neighborhood_radius_df)

	def __convert_food_radius2mm(self):
		fct_convert_in_mm = lambda x, y: round(x / y, 2)
		self.exp.operation_between_2names('food_radius', 'mm2px', fct_convert_in_mm)

	def __plot_batches(self, id_exp, id_ant, batch_list, t0=None, t1=None):
		if len(batch_list) != 0:
			fig, ax = self.__plot_radial_zones(id_exp, id_ant, t0)

			self.__plot_traj(ax, id_exp, id_ant, '0.5', t0, t1)
			self.__plot_mark(ax, id_exp, id_ant, '0.7', t0, t1)
			cols = ColorObject.create_cmap('jet', len(batch_list))
			for i_col, batches in enumerate(batch_list):
				self.__plot_mark(ax, id_exp, id_ant, cols[i_col], batches[0][2], batches[-1][2])
			self.__reorient_plot(ax)

	@staticmethod
	def __reorient_plot(ax):
		for line in ax.lines:
			x_data, y_data = line.get_xdata(), line.get_ydata()
			line.set_xdata(y_data)
			line.set_ydata(x_data)
		ax.invert_xaxis()

	def __plot_traj(self, ax, exp, ant, c, t0=None, t1=None, ms=None):
		if ms is None:
			ms = 5
		if t0 is None:
			t0 = 0
		if t1 is None:
			ax.plot(
				self.exp.x.df.loc[exp, ant, t0:],
				self.exp.y.df.loc[exp, ant, t0:],
				c=c, ms=ms)
		else:
			ax.plot(
				self.exp.x.df.loc[exp, ant, t0:t1],
				self.exp.y.df.loc[exp, ant, t0:t1],
				c=c, ms=ms)

	def __plot_radial_zones(self, exp, ant, t):
		self.exp.load(['x', 'y'])
		fig, ax = plt.subplots(figsize=(5, 6))
		ax.axis('equal')
		ax.set_title('exp:' + str(exp) + ', ant:' + str(ant) + ', t:' + str(t))
		theta = np.arange(-np.pi, np.pi + 0.1, 0.1)
		for radius in [self.arena_radius, self.exp.food_neighborhood_radius.get_value(exp)]:
			ax.plot(radius * np.cos(theta), radius * np.sin(theta), c='grey')
		return fig, ax

	def compute_marking_batch(self, id_exp_list=None):
		id_exp_list = self.exp.set_id_exp_list(id_exp_list)

		self.exp.load(['markings', 'xy_markings', 'r_markings'])

		self.__compute_xy_radial_zones()

		dict_of_idx_exp_ant = self.exp.markings.get_index_dict_of_id_exp_ant()
		array_of_idx_exp_ant = self.exp.markings.get_index_array_of_id_exp_ant()

		self.exp.add_event_extracted_from_timeseries(name_ts='xy_radial_zones', name_extracted_events='zone_change_events')

		batch_interval_list = []
		batch_time_thresh_list = []
		batch_distance_thresh_list = []
		for id_exp in id_exp_list:
			if id_exp in dict_of_idx_exp_ant:
				print(id_exp)
				for id_ant in dict_of_idx_exp_ant[id_exp]:
					if (id_exp, id_ant) in array_of_idx_exp_ant:

						distance_thresh, time_thresh, batch_time_thresh_list, batch_distance_thresh_list = \
							self.__compute_and_add_batch_thresholds(
								id_exp, id_ant, batch_time_thresh_list, batch_distance_thresh_list)

						batch_interval_list = self.__add_ant_batches(id_exp, id_ant, time_thresh, distance_thresh, batch_interval_list)

		self.__write_batch_intervals(batch_interval_list)
		self.__write_batch_threshold(batch_time_thresh_list, batch_distance_thresh_list)

		# plt.show()

	def __write_batch_threshold(self, batch_time_thresh_list, batch_distance_thresh_list):
		self.exp.add_new1d_from_array(
			array=np.array(batch_time_thresh_list, dtype=int),
			name='marking_batch_time_threshold',
			object_type='AntCharacteristics1d', category='Markings',
			label='Marking batch time threshold',
			description='Individual time threshold to define the batches of the marking events'
		)
		self.exp.add_new1d_from_array(
			array=np.array(batch_distance_thresh_list, dtype=int),
			name='marking_batch_distance_threshold',
			object_type='AntCharacteristics1d', category='Markings',
			label='Marking batch distance threshold',
			description='Individual distance threshold to define the batches of the marking events'
		)
		self.exp.write(['marking_batch_time_threshold', 'marking_batch_distance_threshold'])

	def __write_batch_intervals(self, batch_interval_list):
		self.exp.add_new1d_from_array(
			array=np.array(batch_interval_list, dtype=int),
			name='marking_batch_interval',
			object_type='Events1d', category='Markings',
			label='Marking batch interval',
			description='Time intervals between the beginning and the end of the batches of the marking events'
		)
		self.exp.write(['marking_batch_interval'])

	def __compute_and_add_batch_thresholds(self, id_exp, id_ant, batch_time_thresh_list, batch_distance_thresh_list):

		time_thresh = self.__compute_batch_time_threshold(id_exp, id_ant)
		distance_thresh = self.__compute_batch_distance_threshold(id_exp, id_ant)

		batch_time_thresh_list.append([id_exp, id_ant, time_thresh])
		batch_distance_thresh_list.append([id_exp, id_ant, distance_thresh])

		return distance_thresh, time_thresh, batch_time_thresh_list, batch_distance_thresh_list

	def __add_ant_batches(self, id_exp, id_ant, time_thresh, distance_thresh, batch_list):

		ant_zone_change_event_array = self.__get_zone_change_events_for_id_ant(id_exp, id_ant)
		idx_when_ant_enters_food_neighborhood = np.where(ant_zone_change_event_array[:, -1] == 2)[0]

		batch_list = self.__add_batches_during_loops_around_food(
			id_exp, id_ant, time_thresh, distance_thresh,
			idx_when_ant_enters_food_neighborhood,
			ant_zone_change_event_array, batch_list)

		batch_list = self.__add_batches_between_last_food_entry_and_experiment_end(
			id_exp, id_ant, time_thresh, distance_thresh,
			idx_when_ant_enters_food_neighborhood,
			ant_zone_change_event_array, batch_list)

		return batch_list

	def __add_batches_between_last_food_entry_and_experiment_end(
			self, id_exp, id_ant, time_thresh, distance_thresh,
			idx_when_ant_enters_food_neighborhood, ant_zone_change_event_array, batch_interval_list):

		t_last_exit = ant_zone_change_event_array[idx_when_ant_enters_food_neighborhood[-1], 2]

		batch_list = self.__compute_ant_batches(
			id_exp, id_ant, time_thresh, distance_thresh, t_last_exit)

		batch_interval_list = self.__add_batch_intervals(batch_list, batch_interval_list)

		# self.__plot_batches(id_exp, id_ant, batch_list, t_last_exit)

		return batch_interval_list

	def __add_batches_during_loops_around_food(
			self, id_exp, id_ant, time_thresh, distance_thresh,
			idx_when_ant_enters_food_neighborhood, ant_zone_change_event_array, batch_interval_list):

		nbr_entry_in_food_neighborhood = len(idx_when_ant_enters_food_neighborhood)

		for ii in range(nbr_entry_in_food_neighborhood - 1):

			idx_exit = idx_when_ant_enters_food_neighborhood[ii] + 1
			idx_next_entry = idx_when_ant_enters_food_neighborhood[ii + 1]

			list_visited_zones = list(ant_zone_change_event_array[idx_exit:idx_next_entry, -1])
			has_ant_exit_circular_arena = (0 in list_visited_zones)

			if has_ant_exit_circular_arena:
				t_exit = ant_zone_change_event_array[idx_exit, 2]
				t_next_entry = ant_zone_change_event_array[idx_next_entry, 2]

				batch_list = self.__compute_ant_batches(
					id_exp, id_ant, time_thresh, distance_thresh, t_exit, t_next_entry)

				batch_interval_list = self.__add_batch_intervals(batch_list, batch_interval_list)

				# self.__plot_batches(id_exp, id_ant, batch_list, t_exit, t_next_entry)

		return batch_interval_list

	@staticmethod
	def __add_batch_intervals(batch_list, batch_interval_list):
		for batch in batch_list:
			id_exp = batch[0][0]
			id_ant = batch[0][1]
			frame = batch[0][2] - 1
			dt = batch[-1][2] - frame + 1
			batch_interval_list.append([id_exp, id_ant, frame, dt])
		return batch_interval_list

	def __compute_ant_batches(self, id_exp, id_ant, time_thresh, distance_thresh, t0, t1=None):
		xy_mark_array = self.__get_xy_array_of_marking_between_t0_t1(id_exp, id_ant, t0, t1)
		is_there_marking_between_t0_t1 = len(xy_mark_array) != 0
		if is_there_marking_between_t0_t1:
			batch_list = self.__compute_batches_based_on_time_threshold(time_thresh, xy_mark_array)
			batch_list = self.__correct_batches_based_on_distance_threshold(batch_list, distance_thresh)
			return batch_list
		else:
			return []

	def __compute_batches_based_on_time_threshold(self, thresh, xy_mark_array):
		batch_list = []
		current_batch = self.__start_new_batch(0, xy_mark_array)
		for jj in range(1, len(xy_mark_array)):
			is_same_batch = xy_mark_array[jj, 2] - xy_mark_array[jj - 1, 2] < thresh
			if is_same_batch:
				self.__add_to_current_batch(current_batch, jj, xy_mark_array)
			else:
				batch_list = self.__add_current_batch_in_batch_list(batch_list, current_batch)
				current_batch = self.__start_new_batch(jj, xy_mark_array)
		batch_list = self.__add_current_batch_in_batch_list(batch_list, current_batch)
		return batch_list

	def __correct_batches_based_on_distance_threshold(self, batch_list, max_batch_distance):
		batch_list2 = []
		for batch in batch_list:
			idx_when_distance_to_big = self.__find_idx_when_distance_is_to_big(batch, max_batch_distance)
			is_there_distance_to_big = len(idx_when_distance_to_big) != 0
			if is_there_distance_to_big:
				batch_list2 = self.__add_first_sub_batch(batch, batch_list2, idx_when_distance_to_big)
				batch_list2 = self.__add_middle_sub_batches(batch, batch_list2, idx_when_distance_to_big)
				batch_list2 = self.__add_last_sub_batch(batch, batch_list2, idx_when_distance_to_big)
			else:
				batch_list2.append(batch)
		return batch_list2

	@staticmethod
	def __add_last_sub_batch(batch, batch_list2, idx_when_distance_to_big):
		first_idx_last_sub_batch = idx_when_distance_to_big[-1] + 1
		size_last_sub_batch = len(batch) - first_idx_last_sub_batch
		if_last_sub_batch_big_enough = size_last_sub_batch > 2
		if if_last_sub_batch_big_enough:
			batch_list2.append(batch[first_idx_last_sub_batch:])
		return batch_list2

	@staticmethod
	def __add_middle_sub_batches(batch, batch_list2, idx_when_distance_to_big):
		for jj in range(1, len(idx_when_distance_to_big)):
			first_idx_sub_batch = idx_when_distance_to_big[jj - 1] + 1
			last_idx_sub_batch = idx_when_distance_to_big[jj] + 1
			size_sub_batch = last_idx_sub_batch - first_idx_sub_batch
			if_sub_batch_big_enough = size_sub_batch > 2
			if if_sub_batch_big_enough > 2:
				batch_list2.append(batch[first_idx_sub_batch: last_idx_sub_batch])
		return batch_list2

	@staticmethod
	def __add_first_sub_batch(batch, batch_list2, idx_when_distance_to_big):
		size_first_sub_batch = idx_when_distance_to_big[0]
		if_first_sub_batch_big_enough = size_first_sub_batch > 2
		if if_first_sub_batch_big_enough > 3:
			batch_list2.append(batch[:size_first_sub_batch + 1])
		return batch_list2

	@staticmethod
	def __find_idx_when_distance_is_to_big(batch, max_batch_distance):
		batch_xy = np.array(batch)[:, -2:]
		batch_xy_distance = np.around(distance(batch_xy[1:, :], batch_xy[:-1, :]))
		idx_when_distance_to_big = np.where(batch_xy_distance > max_batch_distance)[0]
		return idx_when_distance_to_big

	@staticmethod
	def __start_new_batch(jj, xy_mark_array):
		current_batch = [list(xy_mark_array[jj, :])]
		return current_batch

	@staticmethod
	def __add_current_batch_in_batch_list(batch_list, current_batch):
		is_current_batch_long_enough = len(current_batch) > 2
		if is_current_batch_long_enough:
			batch_list.append(current_batch)
		return batch_list

	@staticmethod
	def __add_to_current_batch(current_batch, jj, xy_mark_array):
		current_batch.append(list(xy_mark_array[jj, :]))

	def __get_xy_array_of_marking_between_t0_t1(self, id_exp, id_ant, t0, t1):
		return np.array(
			self.exp.xy_markings.get_row_id_exp_ant_in_frame_interval(id_exp, id_ant, t0, t1).reset_index())

	def compute_recruitment(self, id_exp_list=None):
		id_exp_list = self.exp.set_id_exp_list(id_exp_list)

		self.exp.load(['r_markings', 'marking_batch_interval'])
		min_lg_rad = 60
		max_lg_rad = 70

		batches_recruitment_intervals = []
		array_batch_interval = self.exp.marking_batch_interval.convert_df_to_array()
		for id_exp, id_ant, batch_frame, batch_dt in array_batch_interval:
			if id_exp in id_exp_list:

				t0 = batch_frame
				t1 = t0 + batch_dt

				r_mark = np.array(self.exp.r_markings.get_row_id_exp_ant_in_frame_interval(id_exp, id_ant, t0, t1))
				rad_min = r_mark.min()
				rad_max = r_mark.max()
				if rad_max - rad_min >= min_lg_rad:
					zone_mark = r_mark[0] - self.arena_radius
					if int(zone_mark) < 0:
						r_mark = np.sort(r_mark, axis=0)
						r_mark = np.array(r_mark[1:] - r_mark[:-1], dtype=int)
						r_mark = r_mark > max_lg_rad
						if np.sum(r_mark) == 0:
							batches_recruitment_intervals.append([id_exp, id_ant, batch_frame, batch_dt])
		#
		# 		batches_recruitment_intervals = self.batch_criterion(
		# 			id_exp, id_ant, batches_recruitment_intervals, thresh, t_exit, t_next_entry)
		#
		# 		t_last_exit = ant_zone_change_event_array[mask_where_ant_enter_food_neighborhood[-1], 2]
		# 		batches_recruitment_intervals = self.batch_criterion(
		# 			id_exp, id_ant, batches_recruitment_intervals, thresh, t_last_exit)
		# 		for ii, batch in enumerate(batches_recruitment_intervals):
		# 			id_ant = int(batch[0, 1])
		# 			t0 = int(batch[0, 2]) - 1
		# 			dt = int(batch[-1, 2]) - t0 + 2
		# 			self.exp.recruitment_interval.add_row((id_exp, id_ant, t0), dt)
		# 			# self.__plot_recruitment_batch(id_exp, batches_recruitment_intervals)
		# 			# plt.show()
		# self.exp.write('recruitment_interval')

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
		self.__reorient_plot(ax)

	def __plot_mark(self, ax, id_exp, id_ant, c, t0=None, t1=None):
		if t0 is None:
			t0 = 0
		if t1 is None:
			ax.plot(
				self.exp.xy_markings.df.loc[IdxSc[id_exp, id_ant, t0:], 'x'],
				self.exp.xy_markings.df.loc[IdxSc[id_exp, id_ant, t0:], 'y'],
				'o', ms=5, c=c)
		else:
			ax.plot(
				self.exp.xy_markings.df.loc[IdxSc[id_exp, id_ant, t0:t1], 'x'],
				self.exp.xy_markings.df.loc[IdxSc[id_exp, id_ant, t0:t1], 'y'],
				'o', ms=5, c=c)

	def __get_zone_change_events_for_id_ant(self, id_exp, id_ant):
		ant_zone_change_event_df = self.exp.zone_change_events.get_row_of_id_exp_ant(id_exp, id_ant)
		ant_zone_change_event_array = np.array(ant_zone_change_event_df.reset_index())
		return ant_zone_change_event_array

	def __compute_batch_time_threshold(self, id_exp, id_ant):
		self.exp.load('marking_interval')
		mark_interval_ant = np.array(self.exp.marking_interval.df.loc[id_exp, id_ant, :].reset_index())
		n_occ, times = np.histogram(mark_interval_ant[:, -1], bins='fd')
		min_thresh = 60
		max_thresh = 200
		thresh1 = self.__compute_batch_threshold(n_occ, times, min_thresh, max_thresh)

		# self.__plot_batch_thresh(id_exp, id_ant, max_thresh, min_thresh, n_occ, thresh1, times, title='time')
		return thresh1

	def __compute_batch_distance_threshold(self, id_exp, id_ant):
		self.exp.load('marking_distance')
		mark_distance_ant = np.array(self.exp.marking_distance.df.loc[id_exp, id_ant, :].reset_index())
		n_occ, times = np.histogram(mark_distance_ant[:, -1], bins='fd')
		min_thresh = 40
		max_thresh = 70
		thresh = self.__compute_batch_threshold(n_occ, times, min_thresh, max_thresh)
		# self.__plot_batch_thresh(id_exp, id_ant, max_thresh, min_thresh, n_occ, thresh, times, title='distance')

		return thresh

	@staticmethod
	def __compute_batch_threshold(n_occ, times, min_thresh, max_thresh):
		mask0 = np.where(n_occ == 0)[0]
		mask1 = np.where(n_occ > 1)[0]
		if len(mask0) == 0:
			thresh = max_thresh
		elif len(mask1) == 0:
			thresh = np.floor(times[mask0[0]])
			if thresh == 0 and len(mask0) > 1:
				thresh = np.floor(times[mask0[1]])
		else:
			times3 = times[mask1[-1]:]
			n_occ2 = n_occ[mask1[-1]:]
			mask0 = np.where(n_occ2 == 0)[0]
			if len(mask0) == 0:
				thresh = np.floor(times3[-1])
			else:
				thresh = np.floor(times3[np.where(n_occ2 == 0)[0][0]])
		thresh = int(max(min(thresh, max_thresh), min_thresh))
		return thresh

	@staticmethod
	def __plot_batch_thresh(id_exp, id_ant, max_thresh, min_thresh, n_occ, thresh1, times, title=''):
		fig, ax = plt.subplots()
		ax.loglog((times[1:] + times[:-1]) / 2., n_occ / np.sum(n_occ), '.-', c='k')
		ax.axvline(thresh1, ls='--', c='black')
		ax.axvline(min_thresh, ls=':', c='grey')
		ax.axvline(max_thresh, ls=':', c='grey')
		ax.set_title(title+', exp:' + str(id_exp) + ', ant:' + str(id_ant))

	def batch_criterion(self, id_exp, id_ant, batches_recruitment, thresh, t0, t1=None):
		min_lg_rad = 60
		max_lg_rad = 70

		xy_mark = self.__get_xy_array_of_marking_between_t0_t1(id_exp, id_ant, t0, t1)
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
					rad_min = self.exp.r.get_row_of_idx_array(batch[:, :3]).min()['r']
					rad_max = self.exp.r.get_row_of_idx_array(batch[:, :3]).max()['r']

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
