import numpy as np
import pandas as pd
import Tools.MiscellaneousTools.Geometry as Geo

from matplotlib import pylab as plt
from pandas import IndexSlice as IdxSc

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.Plotter.ColorObject import ColorObject
from Tools.Plotter.Plotter import Plotter


class AnalyseMarkingBatches(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'MarkingBatch'
        self.arena_radius = 120
        self.min_time_thresh = .6
        self.max_time_thresh = 2.
        self.min_dist_thresh = 30
        self.max_dist_thresh = 50

    def compute_batch_fit(self, redo, plot, redo_hist=False):
        fit_result_name = 'batch_fit_error'
        outside_fit_result_name = 'outside_'+fit_result_name
        inside_fit_result_name = 'inside_'+fit_result_name

        exit_result_name = 'batch_exit_error'
        outside_exit_result_name = 'outside_'+exit_result_name
        inside_exit_result_name = 'inside_'+exit_result_name

        fit_label = '%s batch fit error'
        exit_label = '%s batch exit error'

        fit_description = 'Normalized distance between a linear fit and the markings of a batch done by a %s ants'
        exit_description = 'Angle between the vector (food, exit) ' \
                           'and the linear fit of the markings of a batch done by a %s ants'

        fit_bins = np.arange(-4, 50, 5)
        dtheta = 0.5
        exit_bins = np.arange(0, np.pi+dtheta, dtheta)

        if redo:
            name = 'manual_marking_batches'
            exit_angle_name = 'food_exit_angle'
            marking_name = 'manual_marking_intervals_xy'
            self.exp.load([name, marking_name, exit_angle_name, 'from_outside'])

            xy_name = 'xy'
            self.exp.load_as_2d('x', 'y', xy_name, 'x', 'y')

            batch_array = self.exp.get_df(name).reset_index().values
            batch_array[:, -1] *= 100
            batch_array[:, -1] += batch_array[:, -2]
            batch_array = batch_array.astype(int)

            i = 0
            res_straight_inside = []
            res_err_inside = []

            res_straight_outside = []
            res_err_outside = []
            for id_exp, id_ant, f0, f1 in batch_array:
                print(id_exp, id_ant, f0)
                df = self.exp.get_df(marking_name).loc[id_exp, id_ant, f0:f1].reset_index()
                if len(df) > 4:
                    i += 1
                    df.drop(columns=[id_exp_name, id_ant_name, id_frame_name], inplace=True)
                    df.columns = ['x', 'y']
                    df.set_index('x', inplace=True)
                    df.sort_index(inplace=True)
                    self.exp.add_new_dataset_from_df(df=df, name='temp', category=self.category, replace=True)
                    a, b, x_fit, y_fit = self.exp.fit('temp')
                    straight = round(np.sum(np.abs(df['y'].values-y_fit))/len(df), 2)

                    marking_angle = np.arctan(a)
                    exit_angle = self.exp.get_value(exit_angle_name, (id_exp, f0))
                    marking_error = round(Geo.angle_distance(marking_angle, exit_angle), 3)
                    marking_error = np.abs(min(marking_error, np.pi-marking_error))

                    from_outside = self.exp.get_value('from_outside', (id_exp, id_ant))
                    if from_outside == 1:
                        res_straight_outside.append((id_exp, id_ant, f0, straight))
                        res_err_outside.append((id_exp, id_ant, f0, marking_error))
                        c = 'r'
                    else:
                        res_straight_inside.append((id_exp, id_ant, f0, straight))
                        res_err_inside.append((id_exp, id_ant, f0, marking_error))
                        c = 'b'

                    if plot is True:
                        fig, ax = self.__init_graph(id_exp, id_ant, f0, f1)
                        self.__plot_traj(ax, xy_name, id_exp, id_ant, '0.5', f0, f1)
                        self.__plot_mark(ax, marking_name, id_exp, id_ant, c, f0, f1)
                        self.__reorient_plot(ax)

                        plotter = Plotter(self.exp.root, self.exp.get_data_object('temp'))
                        plotter.plot_fit(preplot=(fig, ax), typ='linear', c='k')
                        ax.set_title(
                            'exp:%i, ant:%i, f:(%i,%i), st.:%.2f, err:%.2f, out:%i' %
                            (id_exp, id_ant, f0, f1, straight, marking_error, from_outside))

                        plotter.save(fig, name=i, sub_folder='marking_batch_fit')

            self.exp.add_new_dataset_from_array(array=np.array(res_straight_inside), name=inside_fit_result_name,
                                                index_names=[id_exp_name, id_ant_name, id_frame_name],
                                                column_names=[inside_fit_result_name], category=self.category,
                                                label=fit_label % 'inside', description=fit_description % 'inside')

            self.exp.add_new_dataset_from_array(array=np.array(res_straight_outside), name=outside_fit_result_name,
                                                index_names=[id_exp_name, id_ant_name, id_frame_name],
                                                column_names=[outside_fit_result_name], category=self.category,
                                                label=fit_label % 'outside', description=fit_description % 'outside')

            self.exp.add_new_dataset_from_array(array=np.array(res_err_inside), name=inside_exit_result_name,
                                                index_names=[id_exp_name, id_ant_name, id_frame_name],
                                                column_names=[inside_exit_result_name], category=self.category,
                                                label=exit_label % 'inside', description=exit_description % 'inside')

            self.exp.add_new_dataset_from_array(array=np.array(res_err_outside), name=outside_exit_result_name,
                                                index_names=[id_exp_name, id_ant_name, id_frame_name],
                                                column_names=[outside_exit_result_name], category=self.category,
                                                label=exit_label % 'outside', description=exit_description % 'outside')

            self.exp.write(inside_fit_result_name)
            self.exp.write(outside_fit_result_name)
            self.exp.write(inside_exit_result_name)
            self.exp.write(outside_exit_result_name)
        else:
            self.exp.load(inside_fit_result_name)
            self.exp.load(outside_fit_result_name)
            self.exp.load(inside_exit_result_name)
            self.exp.load(outside_exit_result_name)

        inside_hist_name = self.compute_hist(inside_fit_result_name, bins=fit_bins, redo=redo, redo_hist=redo_hist)
        outside_hist_name = self.compute_hist(outside_fit_result_name, bins=fit_bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(self.exp.root, self.exp.get_data_object(inside_hist_name))
        fig, ax = plotter.plot(label='inside', normed=True, c='navy')
        plotter = Plotter(self.exp.root, self.exp.get_data_object(outside_hist_name))
        plotter.plot(preplot=(fig, ax), c='r', label='outside', normed=True)
        plotter.save(fig, name=fit_result_name)

        inside_hist_name = self.compute_hist(inside_exit_result_name, bins=exit_bins, redo=redo, redo_hist=redo_hist)
        outside_hist_name = self.compute_hist(outside_exit_result_name, bins=exit_bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(self.exp.root, self.exp.get_data_object(inside_hist_name))
        fig, ax = plotter.plot(label='inside', normed=True, c='navy')
        plotter = Plotter(self.exp.root, self.exp.get_data_object(outside_hist_name))
        plotter.plot(preplot=(fig, ax), c='r', label='outside', normed=True)
        plotter.save(fig, name=exit_result_name)

    def compute_manual_marking_batches(self, redo, redo_hist=False):
        result_name = 'manual_marking_batches'
        label = 'Manual marking batches'
        description = 'Duration of a manual marking batches:' \
                      ' group of markings laid close to each other in time,' \
                      ' Markings has been verified manually'

        bins = np.arange(.1, 20, 1.)

        if redo:
            marking_name = 'manual_marking_intervals'
            marking_xy_name = 'manual_marking_intervals_xy'
            food_distance_name = 'mm10_distance2food'
            self.exp.load([marking_name, marking_xy_name, food_distance_name, 'fps'])

            xy_name = 'xy'
            self.exp.load_as_2d('mm10_x', 'mm10_y', xy_name, 'x', 'y')

            self.__compute_xy_radial_zones(food_distance_name)
            self.exp.event_extraction_from_timeseries(
                name_ts='xy_radial_zones', name_extracted_events='zone_change_events')

            res = []

            def get_batch4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                id_ant = df.index.get_level_values(id_ant_name)[0]
                print(id_exp, id_ant)

                time_thresh = self.__compute_batch_time_threshold(marking_name, id_exp, id_ant)
                dist_thresh = self.__compute_batch_distance_threshold(id_exp, id_ant)

                ant_zone_change_event_array = self.__get_zone_change_events_for_id_ant(id_exp, id_ant)
                if len(ant_zone_change_event_array) != 0:
                    idx_when_ant_enters_food_neighborhood = np.where(ant_zone_change_event_array[:, -1] == 2)[0]

                    self.__add_batches_between_experiment_start_and_first_food_entry(
                        id_exp, id_ant, xy_name, marking_xy_name, time_thresh, dist_thresh,
                        idx_when_ant_enters_food_neighborhood, ant_zone_change_event_array, res)

                    self.__add_batches_during_loops_around_food(
                        id_exp, id_ant, xy_name, marking_xy_name, time_thresh, dist_thresh,
                        idx_when_ant_enters_food_neighborhood, ant_zone_change_event_array, res)

                    self.__add_batches_between_last_food_entry_and_experiment_end(
                        id_exp, id_ant, xy_name, marking_xy_name, time_thresh, dist_thresh,
                        idx_when_ant_enters_food_neighborhood, ant_zone_change_event_array, res)

                    # plt.show()
                return df

            self.exp.groupby(marking_name, [id_exp_name, id_ant_name], get_batch4each_group)

            res = np.array(list(set(res)), dtype=int)
            self.exp.add_new1d_from_array(res, name=result_name, object_type='Events1d',
                                          category=self.category, label=label, description=description)
            self.exp.operation_between_2names(result_name, 'fps', lambda x, y: x/y)
            self.exp.change_df(result_name, self.exp.get_df(result_name).sort_index())

            self.exp.write(result_name)

        hist_name = self.compute_hist(result_name, bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='Marking intervals', xscale='log', yscale='log', ylabel='PDF', normed=True)
        plotter.save(fig)

    def __compute_batch_time_threshold(self, marking_name, id_exp, id_ant):
        mark_interval_ant = np.array(self.exp.get_df(marking_name).loc[id_exp, id_ant, :].reset_index())
        n_occ, times = np.histogram(mark_interval_ant[:, -1], bins='fd')
        thresh1 = self.__compute_batch_threshold(n_occ, times, self.min_time_thresh, self.max_time_thresh)

        # self.__plot_batch_thresh(id_exp, id_ant, min_thresh, max_thresh, n_occ, thresh1, times, title='time')
        return thresh1

    def __compute_batch_distance_threshold(self, id_exp, id_ant):
        self.exp.load('marking_distances')
        mark_distance_ant = np.array(self.exp.get_df('marking_distances').loc[id_exp, id_ant, :].dropna().reset_index())
        n_occ, times = np.histogram(mark_distance_ant[:, -1], bins='fd')
        thresh = self.__compute_batch_threshold(n_occ, times, self.min_dist_thresh, self.max_dist_thresh)
        # self.__plot_batch_thresh(
        #     id_exp, id_ant, self.min_dist_thresh, self.max_dist_thresh, n_occ, thresh, times, title='distance')

        return thresh

    @staticmethod
    def __compute_batch_threshold(n_occ, times, min_thresh, max_thresh):
        mask0 = np.where(n_occ == 0)[0]
        mask1 = np.where(n_occ > 1)[0]
        if len(mask0) == 0:
            thresh = max_thresh
        elif len(mask1) == 0:
            thresh = times[mask0[0]]
            if thresh == 0 and len(mask0) > 1:
                thresh = times[mask0[1]]
        else:
            times3 = times[mask1[-1]:]
            n_occ2 = n_occ[mask1[-1]:]
            mask0 = np.where(n_occ2 == 0)[0]
            if len(mask0) == 0:
                thresh = times3[-1]
            else:
                thresh = times3[np.where(n_occ2 == 0)[0][0]]
        thresh = max(min(thresh, max_thresh), min_thresh)
        return thresh

    @staticmethod
    def __plot_batch_thresh(id_exp, id_ant, min_thresh, max_thresh, n_occ, thresh1, times, title=''):
        fig, ax = plt.subplots()
        ax.loglog((times[1:] + times[:-1]) / 2., n_occ / np.sum(n_occ), '.-', c='k')
        ax.axvline(thresh1, ls='--', c='black')
        ax.axvline(min_thresh, ls=':', c='grey')
        ax.axvline(max_thresh, ls=':', c='grey')
        ax.set_title(title + ', exp:' + str(id_exp) + ', ant:' + str(id_ant))
        plt.show()

    def __get_zone_change_events_for_id_ant(self, id_exp, id_ant):
        try:
            ant_zone_change_event_df = self.exp.zone_change_events.get_row_of_id_exp_ant(id_exp, id_ant)
            ant_zone_change_event_array = np.array(ant_zone_change_event_df.reset_index())
        except KeyError:
            ant_zone_change_event_array = np.zeros([0, 4])
        return ant_zone_change_event_array

    def __compute_xy_radial_zones(self, distance_name):
        self.exp.load('food_radius')

        self.__compute_food_neighborhood_radius()

        self.__compute_is_xy_in_food_neighborhood(distance_name)
        self.__compute_is_xy_in_circular_arena(distance_name)

        self.__compute_sum_of_is_xy_in_food_neighborhood_and_in_circular_arena()

    def __compute_food_neighborhood_radius(self):
        self.exp.add_copy1d(name_to_copy='food_radius', copy_name='food_neighborhood_radius')
        self.exp.operation('food_neighborhood_radius', lambda x: x+15)

    def __compute_is_xy_in_food_neighborhood(self, distance_name):
        self.exp.add_copy1d(name_to_copy=distance_name, copy_name='is_xy_in_food_neighborhood')

        def func_is_lesser_than_circular_arena_radius(x, y):
            return (x - y < 0).astype(int)

        self.exp.operation_between_2names(
            'is_xy_in_food_neighborhood', 'food_neighborhood_radius', func_is_lesser_than_circular_arena_radius)

    def __compute_is_xy_in_circular_arena(self, distance_name):
        self.exp.add_copy1d(distance_name, 'is_in_circular_arena')

        def func_is_lesser_than_arena_radius(x):
            return (x - self.arena_radius < 0).astype(int)

        self.exp.operation('is_in_circular_arena', func_is_lesser_than_arena_radius)

    def __compute_sum_of_is_xy_in_food_neighborhood_and_in_circular_arena(self):
        self.exp.add_copy1d('is_xy_in_food_neighborhood', 'xy_radial_zones')

        self.exp.operation_between_2names('xy_radial_zones', 'is_in_circular_arena', lambda x, y: x+y)

    def __add_batches_during_loops_around_food(
            self, id_exp, id_ant, xy_name, marking_xy_name, time_thresh, dist_thresh,
            idx_when_ant_enters_food_neighborhood, ant_zone_change_event_array, batch_interval_list):

        nbr_entry_in_food_neighborhood = len(idx_when_ant_enters_food_neighborhood)

        for ii in range(nbr_entry_in_food_neighborhood - 1):

            idx_exit = idx_when_ant_enters_food_neighborhood[ii] + 1
            idx_next_entry = idx_when_ant_enters_food_neighborhood[ii + 1]

            # list_visited_zones = list(ant_zone_change_event_array[idx_exit:idx_next_entry, -1])
            # has_ant_exit_circular_arena = (0 in list_visited_zones)

            # if has_ant_exit_circular_arena:
            t_exit = ant_zone_change_event_array[idx_exit, 2]
            t_next_entry = ant_zone_change_event_array[idx_next_entry, 2]

            batch_list = self.__compute_ant_batches(
                id_exp, id_ant, marking_xy_name, time_thresh, dist_thresh, t_exit, t_next_entry)

            self.__add_batch_intervals(batch_list, batch_interval_list)

            # self.__plot_batches_from_batches(
            #     id_exp, id_ant, xy_name, marking_xy_name, batch_list, t_exit, t_next_entry)

    def __compute_ant_batches(self, id_exp, id_ant, marking_xy_name, time_thresh, dist_thresh, t0=None, t1=None):
        xy_mark_array = np.array(
            self.exp.get_data_object(marking_xy_name).get_row_of_id_exp_ant_in_frame_interval(
                id_exp, id_ant, t0, t1).reset_index())
        is_there_marking_between_t0_t1 = len(xy_mark_array) != 0
        if is_there_marking_between_t0_t1:
            batch_list = self.__compute_batches_based_on_time_threshold(time_thresh, xy_mark_array)
            batch_list = self.__correct_batches_based_on_distance_threshold(batch_list, dist_thresh)
            return batch_list
        else:

            return []

    def __compute_batches_based_on_time_threshold(self, thresh, xy_mark_array):
        batch_list = []
        fps = self.exp.get_value('fps', int(xy_mark_array[0, 0]))
        current_batch = [list(xy_mark_array[0, :])]
        for jj in range(1, len(xy_mark_array)):
            is_same_batch = (xy_mark_array[jj, 2] - xy_mark_array[jj - 1, 2])/fps < thresh
            if is_same_batch:
                current_batch.append(list(xy_mark_array[jj, :]))
            else:
                batch_list = self.__add_current_batch_in_batch_list(batch_list, current_batch)
                current_batch = [list(xy_mark_array[jj, :])]
        batch_list = self.__add_current_batch_in_batch_list(batch_list, current_batch)
        return batch_list

    @staticmethod
    def __add_current_batch_in_batch_list(batch_list, current_batch):
        is_current_batch_long_enough = len(current_batch) > 2
        if is_current_batch_long_enough:
            batch_list.append(current_batch)
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
    def __find_idx_when_distance_is_to_big(batch, max_batch_distance):
        batch_xy = np.array(batch)[:, -2:]
        batch_xy_distance = np.around(Geo.distance(batch_xy[1:, :], batch_xy[:-1, :]))
        idx_when_distance_to_big = np.where(batch_xy_distance > max_batch_distance)[0]
        return idx_when_distance_to_big

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
    def __add_last_sub_batch(batch, batch_list2, idx_when_distance_to_big):
        first_idx_last_sub_batch = idx_when_distance_to_big[-1] + 1
        size_last_sub_batch = len(batch) - first_idx_last_sub_batch
        if_last_sub_batch_big_enough = size_last_sub_batch > 2
        if if_last_sub_batch_big_enough:
            batch_list2.append(batch[first_idx_last_sub_batch:])
        return batch_list2

    @staticmethod
    def __add_first_sub_batch(batch, batch_list2, idx_when_distance_to_big):
        size_first_sub_batch = idx_when_distance_to_big[0]
        is_first_sub_batch_big_enough = size_first_sub_batch > 2
        if is_first_sub_batch_big_enough:
            batch_list2.append(batch[:size_first_sub_batch + 1])
        return batch_list2

    @staticmethod
    def __add_batch_intervals(batch_list, batch_interval_list):
        for batch in batch_list:
            id_exp = batch[0][0]
            id_ant = batch[0][1]
            frame = batch[0][2] - 1
            dt = batch[-1][2] - frame + 1
            batch_interval_list.append((id_exp, id_ant, frame, dt))

    def __plot_batches_from_batches(self, id_exp, id_ant, xy_name, marking_name, batch_list, t0=None, t1=None):
        if len(batch_list) != 0:
            fig, ax = self.__init_graph(id_exp, id_ant, t0, t1)
            self.__plot_traj(ax, xy_name, id_exp, id_ant, '0.5', t0, t1)
            self.__plot_mark(ax, marking_name, id_exp, id_ant, '0.7', t0, t1)

            cols = ColorObject.create_cmap('jet', len(batch_list))
            for i_col, batches in enumerate(batch_list):
                self.__plot_mark(ax, marking_name, id_exp, id_ant, cols[str(i_col)], batches[0][2], batches[-1][2])
            self.__reorient_plot(ax)

    @staticmethod
    def __init_graph(id_exp, id_ant, t0, t1):
        fig, ax = plt.subplots(figsize=(5, 6))
        ax.axis('equal')
        if t0 is None:
            t02 = 0
        else:
            t02 = t0
        if t1 is None:
            t12 = 0
        else:
            t12 = t1
        ax.set_title('exp: %i, ant: %i, t: (%i, %i)' % (id_exp, id_ant, t02, t12))
        return fig, ax

    def __plot_traj(self, ax, xy_name, exp, ant, c=None, t0=None, t1=None, ms=None):
        if c is None:
            c = 'grey'
        if ms is None:
            ms = 5
        if t0 is None:
            t0 = 0
        if t1 is None:
            ax.plot(
                self.exp.get_df(xy_name).loc[exp, ant, t0:]['x'],
                self.exp.get_df(xy_name).loc[exp, ant, t0:]['y'],
                c=c, ms=ms)
        else:
            ax.plot(
                self.exp.get_df(xy_name).loc[exp, ant, t0:t1]['x'],
                self.exp.get_df(xy_name).loc[exp, ant, t0:t1]['y'],
                c=c, ms=ms)

    def __plot_mark(self, ax, marking_name, id_exp, id_ant=None, c=None, t0=None, t1=None):
        if c is None:
            c = '0.7'
        if t0 is None:
            t0 = 0

        if t1 is None:
            xy_mark = self.exp.get_df(marking_name).loc[IdxSc[id_exp, :, t0:], :]
        else:
            xy_mark = self.exp.get_df(marking_name).loc[IdxSc[id_exp, :, t0:t1], :]
        if id_ant is not None:
            xy_mark = xy_mark.loc[IdxSc[:, id_ant, :], :]

        ax.plot(xy_mark[xy_mark.columns[0]], xy_mark[xy_mark.columns[1]], 'o', ms=5, c=c)

    @staticmethod
    def __reorient_plot(ax):
        # for line in ax.lines:
        #     x_data, y_data = line.get_xdata(), line.get_ydata()
        #     line.set_xdata(y_data)
        #     line.set_ydata(x_data)
        ax.invert_xaxis()

    def __add_batches_between_experiment_start_and_first_food_entry(
            self, id_exp, id_ant, xy_name, marking_xy_name, time_thresh, dist_thresh,
            idx_when_ant_enters_food_neighborhood, ant_zone_change_event_array, batch_interval_list):

        if len(idx_when_ant_enters_food_neighborhood) != 0:
            idx_first_entry = idx_when_ant_enters_food_neighborhood[0]
            if idx_first_entry == 0:
                t_first_entry = 0
            else:
                t_first_entry = ant_zone_change_event_array[idx_first_entry, 2]

            batch_list = self.__compute_ant_batches(
                id_exp, id_ant, marking_xy_name, time_thresh, dist_thresh, t1=t_first_entry)

            self.__add_batch_intervals(batch_list, batch_interval_list)

            # self.__plot_batches_from_batches(id_exp, id_ant, xy_name, marking_xy_name, batch_list, t1=t_first_entry)

    def __add_batches_between_last_food_entry_and_experiment_end(
            self, id_exp, id_ant, xy_name, marking_xy_name, time_thresh, dist_thresh,
            idx_when_ant_enters_food_neighborhood, ant_zone_change_event_array, batch_interval_list):

        if len(idx_when_ant_enters_food_neighborhood) == 0:
            t_last_exit = None
        else:
            idx_last_exit = idx_when_ant_enters_food_neighborhood[-1]
            if idx_last_exit == len(ant_zone_change_event_array)-1:
                t_last_exit = None
            else:
                t_last_exit = ant_zone_change_event_array[idx_last_exit + 1, 2]

        batch_list = self.__compute_ant_batches(id_exp, id_ant, marking_xy_name, time_thresh, dist_thresh, t_last_exit)

        self.__add_batch_intervals(batch_list, batch_interval_list)

        # self.__plot_batches_from_batches(id_exp, id_ant, xy_name, marking_xy_name, batch_list, t_last_exit)
