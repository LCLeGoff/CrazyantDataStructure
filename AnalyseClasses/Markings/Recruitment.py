import numpy as np

from matplotlib import pylab as plt
from pandas import IndexSlice as IdxSc

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from Tools.MiscellaneousTools.Geometry import distance
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager
from Tools.Plotter.ColorObject import ColorObject


class Recruitment(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.pd_idx_manager = PandasIndexManager()
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

        def func_sum(x, y):
            return x + y

        self.exp.operation_between_2names('xy_radial_zones', 'is_in_circular_arena', func_sum)

    def __compute_is_xy_in_circular_arena(self):
        self.exp.add_copy1d('r', 'is_in_circular_arena')

        def func_is_lesser_than_arena_radius(x):
            return (x - self.arena_radius < 0).astype(int)

        self.exp.operation('is_in_circular_arena', func_is_lesser_than_arena_radius)

    def __compute_is_xy_in_food_neighborhood(self):
        self.exp.add_copy1d(name_to_copy='r', copy_name='is_xy_in_food_neighborhood')

        def func_is_lesser_than_circular_arena_radius(x, y):
            return (x - y < 0).astype(int)

        self.exp.operation_between_2names(
            'is_xy_in_food_neighborhood', 'food_neighborhood_radius', func_is_lesser_than_circular_arena_radius)

    def __compute_food_neighborhood_radius(self):
        self.exp.add_copy1d(
            name_to_copy='food_radius', copy_name='food_neighborhood_radius',
            category='Arena', label='food neighborhood radius',
            description='Radius of the food neighborhood'
        )
        food_neighborhood_radius_df = self.exp.food_radius.get_column_values() + 15
        self.exp.food_neighborhood_radius.replace_values(food_neighborhood_radius_df)
        self.exp.write('food_neighborhood_radius')

    def __convert_food_radius2mm(self):
        def func_convert_in_mm(x, y):
            return round(x / y, 2)

        self.exp.operation_between_2names('food_radius', 'mm2px', func_convert_in_mm)

    def __plot_batches_from_batches(self, id_exp, id_ant, batch_list, t0=None, t1=None):
        if len(batch_list) != 0:
            fig, ax = self.__init_graph(id_exp, id_ant, t0)
            self.__plot_radial_zones(ax, id_exp)
            self.__plot_traj(ax, id_exp, id_ant, '0.5', t0, t1)
            self.__plot_mark(ax, id_exp, id_ant, '0.7', t0, t1)

            cols = ColorObject.create_cmap('jet', len(batch_list))
            for i_col, batches in enumerate(batch_list):
                self.__plot_mark(ax, id_exp, id_ant, cols[i_col], batches[0][2], batches[-1][2])
            self.__reorient_plot(ax)

    def __plot_batches_from_batch_interval(self, id_exp, id_ant, batch_frame, batch_dt):
        t0 = batch_frame
        t1 = t0 + batch_dt
        fig, ax = self.__init_graph(id_exp, id_ant, t0)
        self.__plot_arena(ax)
        self.__plot_traj(ax, id_exp, id_ant, '0.5', t0, t1)
        self.__plot_mark(ax, id_exp, id_ant, '0.7', t0, t1)
        self.__reorient_plot(ax)

    @staticmethod
    def __reorient_plot(ax):
        for line in ax.lines:
            x_data, y_data = line.get_xdata(), line.get_ydata()
            line.set_xdata(y_data)
            line.set_ydata(x_data)
        ax.invert_xaxis()

    def __plot_traj(self, ax, exp, ant, c=None, t0=None, t1=None, ms=None):
        self.exp.load(['x', 'y'])
        if c is None:
            c = 'grey'
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

    @staticmethod
    def __init_graph(id_exp, id_ant, t):
        fig, ax = plt.subplots(figsize=(5, 6))
        ax.axis('equal')
        ax.set_title('exp:' + str(id_exp) + ', ant:' + str(id_ant) + ', t:' + str(t))
        return fig, ax

    def __plot_arena(self, ax):
        radius = self.arena_radius
        self.__plot_circle(ax, radius)

    @staticmethod
    def __plot_circle(ax, radius):
        theta = np.arange(-np.pi, np.pi + 0.1, 0.1)
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), c='grey')

    def __plot_radial_zones(self, ax, id_exp):
        for radius in [self.arena_radius, self.exp.food_neighborhood_radius.get_value(id_exp)]:
            self.__plot_circle(ax, radius)

    def compute_marking_batch(self, id_exp_list=None):
        id_exp_list = self.exp.set_id_exp_list(id_exp_list)

        self.exp.load(['markings', 'xy_markings', 'r_markings'])

        self.__compute_xy_radial_zones()

        dict_of_idx_exp_ant = self.exp.markings.get_index_dict_of_id_exp_ant()
        array_of_idx_exp_ant = self.exp.markings.get_index_array_of_id_exp_ant()

        self.exp.event_extraction_from_timeseries(name_ts='xy_radial_zones', name_extracted_events='zone_change_events')

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

                        batch_interval_list = self.__add_ant_batches(id_exp, id_ant, time_thresh, distance_thresh,
                                                                     batch_interval_list)

            plt.show()
        self.__write_batch_intervals(batch_interval_list)
        self.__write_batch_threshold(batch_time_thresh_list, batch_distance_thresh_list)

    def __write_batch_threshold(self, batch_time_thresh_list, batch_distance_thresh_list):
        self.exp.add_new1d_from_array(
            array=np.array(batch_time_thresh_list, dtype=int),
            name='marking_batch_time_thresholds',
            object_type='AntCharacteristics1d', category='Markings',
            label='Marking batch time threshold',
            description='Individual time threshold to define the batches of the marking events'
        )
        self.exp.add_new1d_from_array(
            array=np.array(batch_distance_thresh_list, dtype=int),
            name='marking_batch_distance_thresholds',
            object_type='AntCharacteristics1d', category='Markings',
            label='Marking batch distance threshold',
            description='Individual distance threshold to define the batches of the marking events'
        )
        self.exp.write(['marking_batch_time_thresholds', 'marking_batch_distance_thresholds'])

    def __write_batch_intervals(self, batch_interval_list):
        self.exp.add_new1d_from_array(
            array=np.array(batch_interval_list, dtype=int),
            name='marking_batch_intervals',
            object_type='Events1d', category='Markings',
            label='Marking batch interval',
            description='Time intervals between the beginning and the end of the batches of the marking events'
        )
        self.exp.write(['marking_batch_intervals'])

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

        self.__plot_batches_from_batches(id_exp, id_ant, batch_list, t_last_exit)

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

                self.__plot_batches_from_batches(id_exp, id_ant, batch_list, t_exit, t_next_entry)

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
        is_first_sub_batch_big_enough = size_first_sub_batch > 2
        if is_first_sub_batch_big_enough:
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
            self.exp.xy_markings.get_row_of_id_exp_ant_in_frame_interval(id_exp, id_ant, t0, t1).reset_index())

    def compute_recruitment(self, id_exp_list=None):
        name = 'recruitment_intervals'
        id_exp_list = self.exp.set_id_exp_list(id_exp_list)

        self.exp.load(['r_markings', 'xy_markings', 'marking_batch_intervals'])
        min_batch_r_length = 60

        batches_recruitment_intervals = []
        array_batch_interval = self.exp.marking_batch_intervals.convert_df_to_array()
        for id_exp, id_ant, batch_frame, batch_dt in array_batch_interval:
            if id_exp in id_exp_list:

                r_mark = self.__get_r_from_marking_batch_interval(id_exp, id_ant, batch_frame, batch_dt)
                r_min = r_mark.min()
                r_max = r_mark.max()
                is_batch_long_enough = r_max - r_min >= min_batch_r_length
                if is_batch_long_enough:
                    idx_when_ant_exits_circular_arena = np.where(r_mark > self.arena_radius)[0]
                    does_ant_exits_circular_arena = len(idx_when_ant_exits_circular_arena) != 0
                    batch_length = len(r_mark)
                    if does_ant_exits_circular_arena:
                        idx_first_exit = idx_when_ant_exits_circular_arena[0]
                    else:
                        idx_first_exit = batch_length
                    is_at_least_3_first_marking_inside_circular_arena = idx_first_exit > 2
                    is_first_third_batch_inside_circular_arena = idx_first_exit / float(batch_length) > 1 / 3.
                    if is_at_least_3_first_marking_inside_circular_arena or is_first_third_batch_inside_circular_arena:
                        batches_recruitment_intervals.append([id_exp, id_ant, batch_frame, batch_dt])

        self.__plot_recruitment_batch(batches_recruitment_intervals)

        batches_recruitment_intervals = np.array(batches_recruitment_intervals, dtype=int)
        self.exp.add_new1d_from_array(
            array=batches_recruitment_intervals,
            name=name, object_type='Events1d', category='Markings',
            label='Recruitment time interval',
            description='Time intervals of the recruitment events'
        )
        self.exp.write(name)

    def __get_r_from_marking_batch_interval(self, id_exp, id_ant, batch_frame, batch_dt):
        t0 = batch_frame
        t1 = t0 + batch_dt
        r_mark = self.exp.r_markings.get_row_of_id_exp_ant_in_frame_interval(id_exp, id_ant, t0, t1)
        return np.array(r_mark)

    def __plot_recruitment_batch(self, recruitment_batches, id_exp=None, show=True):

        recruitment_batches_array = np.array(recruitment_batches)
        if id_exp is None:
            list_id_exp = sorted(set(recruitment_batches_array[:, 0]))
        else:
            list_id_exp = [id_exp]

        for id_exp in list_id_exp:
            idx_where_it_is_id_exp = recruitment_batches_array[:, 0] == id_exp
            exp_batches = recruitment_batches_array[idx_where_it_is_id_exp, :]

            fig, ax = self.__init_graph(id_exp, '', '')
            self.__plot_arena(ax)
            self.__plot_mark(ax, id_exp)

            t_first_batch = np.inf
            first_batch = []
            cols = ColorObject.create_cmap('jet', len(exp_batches))
            for i, batch2plot in enumerate(exp_batches):
                id_ant = batch2plot[1]
                t0 = batch2plot[2]
                if t0 < t_first_batch:
                    first_batch = batch2plot
                    t_first_batch = t0
                self.__plot_mark(ax, id_exp, id_ant, cols[i], t0, t0 + batch2plot[3])
            if len(first_batch) != 0:
                self.__plot_mark(ax, id_exp, first_batch[1], 'y', t_first_batch, t_first_batch + first_batch[3])
            self.__reorient_plot(ax)
            if show:
                plt.show()

    def __plot_mark(self, ax, id_exp, id_ant=None, c=None, t0=None, t1=None):
        self.exp.load(['xy_markings'])
        if c is None:
            c = '0.7'
        if t0 is None:
            t0 = 0

        if t1 is None:
            xy_mark = self.exp.xy_markings.df.loc[IdxSc[id_exp, :, t0:], :]
        else:
            xy_mark = self.exp.xy_markings.df.loc[IdxSc[id_exp, :, t0:t1], :]
        if id_ant is not None:
            xy_mark = xy_mark.loc[IdxSc[:, id_ant, :], :]

        ax.plot(xy_mark['x'], xy_mark['y'], 'o', ms=5, c=c)

    def __get_zone_change_events_for_id_ant(self, id_exp, id_ant):
        ant_zone_change_event_df = self.exp.zone_change_events.get_row_of_id_exp_ant(id_exp, id_ant)
        ant_zone_change_event_array = np.array(ant_zone_change_event_df.reset_index())
        return ant_zone_change_event_array

    def __compute_batch_time_threshold(self, id_exp, id_ant):
        self.exp.load('marking_intervals')
        mark_interval_ant = np.array(self.exp.marking_intervals.df.loc[id_exp, id_ant, :].reset_index())
        n_occ, times = np.histogram(mark_interval_ant[:, -1], bins='fd')
        min_thresh = 60
        max_thresh = 200
        thresh1 = self.__compute_batch_threshold(n_occ, times, min_thresh, max_thresh)

        self.__plot_batch_thresh(id_exp, id_ant, max_thresh, min_thresh, n_occ, thresh1, times, title='time')
        return thresh1

    def __compute_batch_distance_threshold(self, id_exp, id_ant):
        self.exp.load('marking_distances')
        mark_distance_ant = np.array(self.exp.marking_distances.df.loc[id_exp, id_ant, :].reset_index())
        n_occ, times = np.histogram(mark_distance_ant[:, -1], bins='fd')
        min_thresh = 40
        max_thresh = 70
        thresh = self.__compute_batch_threshold(n_occ, times, min_thresh, max_thresh)
        self.__plot_batch_thresh(id_exp, id_ant, max_thresh, min_thresh, n_occ, thresh, times, title='distance')

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
        ax.set_title(title + ', exp:' + str(id_exp) + ', ant:' + str(id_ant))
