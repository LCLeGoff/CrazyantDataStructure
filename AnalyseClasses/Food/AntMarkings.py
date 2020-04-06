import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_frame_name, id_ant_name

import Tools.MiscellaneousTools.Geometry as Geo
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager
from Tools.Plotter.Plotter import Plotter


class AnalyseAntMarkings(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.pd_idx_manager = PandasIndexManager()
        self.category = 'AntMarkings'

    @staticmethod
    def get_sequence(tab, dt=5):
        tab2 = tab.ravel()
        res = list(argrelextrema(tab2, np.less)[0])
        res += list(argrelextrema(tab2, np.greater)[0])
        res.sort()
        res = np.array(res)

        dx = res[1:] - res[:-1]
        mask = np.where(dx < dt)[0]
        while len(mask) != 0:
            ii = mask[0]
            if len(mask) > 1 and mask[1] == ii + 1:
                new_x = int(np.mean(res[[ii, ii + 2]]))
                res = np.delete(res, [ii, ii + 1, ii + 2])
                res = np.append(res, new_x)
                res = np.sort(res)
            else:
                res = np.delete(res, [ii, ii + 1])
            dx = res[1:] - res[:-1]
            mask = np.where(dx < dt)[0]
        res = list(set([0] + list(res) + [len(tab2) - 1]))
        res.sort()
        return res

    def compute_potential_markings(self, redo=False):
        result_name = 'potential_markings'
        label = 'Markings'
        description = 'If the ant is potentially marking'

        if redo is True:

            name_speed = 'speed'
            name_orientation = 'speed_phi'
            name_radius = 'food_radius'
            name_fps = 'fps'
            self.exp.load([name_speed, name_orientation, name_radius, name_fps])
            name_xy = 'xy'
            self.exp.load_as_2d('mm10_x', 'mm10_y', name_xy, 'x', 'y', replace=True)

            name_food_xy = 'food_xy'
            self.exp.load_as_2d('mm10_food_x', 'mm10_food_y', name_food_xy, 'x', 'y', replace=True)

            self.exp.add_copy(old_name='mm10_x', new_name=result_name, category=self.category,
                              label=label, description=description)
            self.exp.get_data_object(result_name).df[:] = 0
            self.exp.get_data_object(result_name).df = self.exp.get_df(result_name).astype(int)

            dv_min = 2
            v_max = 5
            dtheta_max = 90 * np.pi / 180
            min_dist = 2

            def get_makings4each_group(df: pd.DataFrame):

                id_exp = df.index.get_level_values(id_exp_name)[0]
                fps = self.exp.get_value(name_fps, id_exp)
                if len(df) > 2*fps:
                    id_ant = df.index.get_level_values(id_ant_name)[0]

                    print(id_exp, id_ant)
                    radius = self.exp.get_value(name_radius, id_exp)

                    df_speed0 = df.loc[id_exp, id_ant, :].copy() / 10.
                    df_orient0 = self.exp.get_df(name_orientation).loc[id_exp, id_ant, :].copy() / 100.

                    speed_tab = df_speed0.dropna().reset_index().values
                    orient_df = df_orient0.loc[id_exp, id_ant, :].dropna()

                    orient_frames = set(orient_df.index)

                    xs = np.array(self.get_sequence(speed_tab[:, 1], dt=5))
                    if len(xs) > 3:
                        a_speed, _ = Geo.get_line_tab(speed_tab[xs[:-1], :], speed_tab[xs[1:], :])

                        v = speed_tab[xs, :]
                        dv = v[1:, 1] - v[:-1, 1]
                        mask_a = a_speed[:-1] < 0
                        mask_dv = (dv[:-1] < -dv_min) * (dv[1:] > dv_min)
                        mask_v = v[1:-1, 1] < v_max
                        mask = list(np.where(mask_a*mask_dv*mask_v)[0] + 1)

                        for ii in mask:
                            f0, f1, f2 = v[ii - 1:ii + 2, 0].astype(int)
                            if f2 - f0 < 32:
                                if {f0, f1, f2}.issubset(orient_frames):
                                    theta0, theta1, theta2 = orient_df.loc[[f0, f1, f2]].values.ravel()
                                    dtheta = np.abs(Geo.angle_distance(theta0, theta2))

                                    if dtheta < dtheta_max:
                                        xy = self.exp.get_df(name_xy).loc[id_exp, id_ant, f1]
                                        if self.exp.get_index(name_food_xy).isin([(id_exp, f1)]).any():
                                            xy_food = self.exp.get_df(name_food_xy).loc[id_exp, f1]
                                            dist_ant_food = Geo.distance_df(xy_food, xy)
                                            if  dist_ant_food > radius+5:

                                                xys = self.exp.get_df(name_xy).loc[id_exp, :, f1]
                                                dist_ant_other_ants = Geo.squared_distance_df(xys, xy)
                                                dist_ant_other_ants =\
                                                    dist_ant_other_ants.drop((id_exp, id_ant, f1)).min() / 10.
                                                if dist_ant_other_ants > min_dist:
                                                    self.exp.change_value(result_name, (id_exp, id_ant, f1), 1)
                                        else:
                                            xys = self.exp.get_df(name_xy).loc[id_exp, :, f1]
                                            dist_ant_other_ants = Geo.squared_distance_df(xys, xy)
                                            dist_ant_other_ants = \
                                                dist_ant_other_ants.drop((id_exp, id_ant, f1)).min() / 10.
                                            if dist_ant_other_ants > min_dist:
                                                self.exp.change_value(result_name, (id_exp, id_ant, f1), 1)
                return df

            self.exp.groupby(name_speed, [id_exp_name, id_ant_name], get_makings4each_group)

            self.exp.write(result_name)

    def compute_potential_marking_intervals(self, redo=False, redo_hist=False):

        result_name = 'potential_marking_intervals'

        bins = np.arange(0.1, 1e3, .1)
        if redo is True:
            name = 'potential_markings'
            self.exp.load(name)

            self.exp.change_df(name, 1-self.exp.get_df(name))
            self.exp.compute_time_intervals(name_to_intervals=name, category=self.category,
                                            result_name=result_name, label='Potential marking time intervals',
                                            description='Time intervals between two potential markings (s)')

            self.exp.write(result_name)

        hist_name = self.compute_hist(result_name, bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='Marking intervals', xscale='log', yscale='log', ylabel='PDF', ls='', normed=True)
        plotter.save(fig)

    def compute_potential_marking_intervals_xy(self):
        result_name = 'potential_marking_intervals_xy'

        marking_name = 'potential_marking_intervals'
        self.exp.load([marking_name])

        name_xy = 'xy'
        self.exp.load_as_2d('mm10_x', 'mm10_y', name_xy, 'x', 'y', replace=True)

        df = self.exp.get_df(name_xy)
        df = df.reindex(self.exp.get_index(marking_name))

        self.exp.add_new2d_from_df(df=df, name=result_name, object_type='Events2d',
                                   xname='marking_x', yname='marking_y',
                                   category=self.category, label='Coordinates of the markings (mm)',
                                   xlabel='X coordinates of the markings (mm)',
                                   ylabel='Y coordinates of the markings (mm)',
                                   description='Coordinates of the markings (mm)')

        self.exp.write(result_name)

    def compute_marking_repartition(self, redo=False, redo_hist=False):
        outside_result_name = 'outside_marking_repartition'
        inside_result_name = 'inside_marking_repartition'

        dtheta = np.pi/10.
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta/2., dtheta)

        if redo:
            label = '%s marking repartition'
            description = 'Angle between the food-exit vector and the food-marking vector from %s ants'

            marking_name = 'marking_intervals'
            food_exit_angle_name = 'food_exit_angle'
            first_frame_name = 'first_attachment_time_of_outside_ant'
            self.exp.load([marking_name, food_exit_angle_name, first_frame_name, 'from_outside'])

            name_xy = 'xy'
            self.exp.load_as_2d('mm10_x', 'mm10_y', name_xy, 'x', 'y', replace=True)

            name_food_xy = 'food_xy'
            self.exp.load_as_2d('mm10_food_x', 'mm10_food_y', name_food_xy, 'x', 'y', replace=True)

            res_outside = []
            res_inside = []

            for id_exp, id_ant, frame in self.exp.get_index(marking_name):
                print(id_exp, id_ant, frame)
                first_frame = self.exp.get_value(first_frame_name, id_exp)
                if frame > first_frame and self.exp.get_index(name_food_xy).isin([(id_exp, frame)]).any():
                    xy = self.exp.get_df(name_xy).loc[(id_exp, id_ant, frame)]
                    xy_food = self.exp.get_df(name_food_xy).loc[(id_exp, frame)]
                    exit_angle = self.exp.get_value(food_exit_angle_name, (id_exp, frame))

                    ant_angle = Geo.angle_df(xy_food, xy)

                    angle = Geo.angle_distance(ant_angle, exit_angle)
                    from_outside = self.exp.get_value('from_outside', (id_exp, id_ant))
                    if from_outside == 1:
                        res_outside.append((id_exp, id_ant, frame, angle))
                    else:
                        res_inside.append((id_exp, id_ant, frame, angle))

            res_outside = np.array(res_outside)
            res_inside = np.array(res_inside)

            self.exp.add_new1d_from_array(array=res_outside, name=outside_result_name, object_type='Events1d',
                                          category=self.category,
                                          label=label % 'outside', description=description % 'outside')

            self.exp.add_new1d_from_array(array=res_inside, name=inside_result_name, object_type='Events1d',
                                          category=self.category,
                                          label=label % 'inside', description=description % 'inside')

            self.exp.write([outside_result_name, inside_result_name])
        else:
            self.exp.load([outside_result_name, inside_result_name])

        outside_hist_name = self.compute_hist(name=outside_result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        inside_hist_name = self.compute_hist(name=inside_result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        title = 'Marking repartition'
        xlabel = 'Marking-food-exit angle (rad)'
        graph_name = 'marking_repartition'
        self._plot_outside_and_inside_hist(graph_name, inside_hist_name, outside_hist_name, title, xlabel)

    def outside_marking_repartition_evol(self, redo):
        label_detail = 'outside'
        name = '%s_marking_repartition' % label_detail
        result_name = name + '_hist_evol'

        dtheta = np.pi/10.
        bins = np.arange(0, np.pi+dtheta/2., dtheta)

        dx = 0.25
        dx2 = 0.25
        start_frame_intervals = np.arange(0, 3., dx)*60*100
        end_frame_intervals = start_frame_intervals+dx2*60*100

        if redo:
            label = '%s marking repartition over time (rad)'
            description = 'Angle between the food-exit vector and the food-marking vector from %s ants over time'

            init_frame_name = 'first_attachment_time_of_outside_ant'
            self.exp.load([name, init_frame_name])

            self.change_first_frame(name, init_frame_name)
            self.exp.operation(name, lambda x: np.abs(x))
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                      end_frame_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category,
                                      label=label % label_detail, description=description % label_detail)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Marking-food-exit angle (rad)', ylabel='PDF',
                               normed=True, label_suffix='s')
        ax.set_ylim(0, 2)
        plotter.save(fig)

    def inside_marking_repartition_evol(self, redo):
        label_detail = 'inside'
        name = '%s_marking_repartition' % label_detail
        result_name = name + '_hist_evol'

        dtheta = np.pi/10.
        bins = np.arange(0, np.pi+dtheta/2., dtheta)

        dx = 0.25
        dx2 = 0.25
        start_frame_intervals = np.arange(0, 3., dx)*60*100
        end_frame_intervals = start_frame_intervals+dx2*60*100

        if redo:
            label = '%s marking repartition over time (rad)'
            description = 'Angle between the food-exit vector and the food-marking vector from %s ants over time'

            init_frame_name = 'first_attachment_time_of_outside_ant'
            self.exp.load([name, init_frame_name])

            self.change_first_frame(name, init_frame_name)
            self.exp.operation(name, lambda x: np.abs(x))
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                      end_frame_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category,
                                      label=label % label_detail, description=description % label_detail)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Marking-food-exit angle (rad)', ylabel='PDF',
                               normed=True, label_suffix='s')
        ax.set_ylim(0, 2)
        plotter.save(fig)

    def _plot_outside_and_inside_hist(
            self, graph_name, inside_hist_name, outside_hist_name, title, xlabel,
            normed=True, xscale=None, yscale=None):

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(outside_hist_name))
        fig, ax = plotter.create_plot()
        plotter.plot(preplot=(fig, ax), xscale=xscale, yscale=yscale, normed=normed, label='outside', c='r')
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(inside_hist_name))
        plotter.plot(preplot=(fig, ax), xscale=xscale, yscale=yscale, normed=normed, label='inside', c='navy')
        plotter.draw_legend(ax)
        plotter.display_title(ax, title=title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('PDF')
        plotter.save(fig, name=graph_name)

    def compute_markings_time_distance(self, redo=False, redo_hist=False):
        outside_result_name = 'outside_markings_time_distance'
        inside_result_name = 'inside_markings_time_distance'

        dt = 0.5
        bins = list(np.arange(0, 20, dt))

        if redo:

            marking_name = 'manual_marking_intervals'
            self.exp.load([marking_name, 'fps', 'from_outside'])

            label = 'Time distance between %s markings'
            description = 'Distance in time between the actual %s marking and the previous marking'

            outside_res = []
            inside_res = []

            def get_distance4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                fps = self.exp.get_value('fps', id_exp)

                frames = list(df.index.get_level_values(id_frame_name))
                frames.sort()

                arr = df.loc[id_exp, :].reset_index().values
                for id_ant, frame, _ in arr:
                    from_outside = self.exp.get_value('from_outside', (id_exp, id_ant))
                    mask = np.where(frames < frame)[0]
                    if len(mask) == 0:
                        dist = frame-frames[1]
                    else:
                        dist = frame-frames[mask[-1]]
                    dist /= fps
                    if from_outside == 1:
                        outside_res.append((id_exp, id_ant, frame, dist))
                    else:
                        inside_res.append((id_exp, id_ant, frame, dist))

            self.exp.groupby(marking_name, id_exp_name, get_distance4each_group)

            self.exp.add_new1d_from_array(array=np.array(outside_res), name=outside_result_name, object_type='Events1d',
                                          category=self.category, label=label % 'outside',
                                          description=description % 'outside')

            self.exp.add_new1d_from_array(array=np.array(inside_res), name=inside_result_name, object_type='Events1d',
                                          category=self.category, label=label % 'inside',
                                          description=description % 'inside')

            self.exp.write(outside_result_name)
            self.exp.write(inside_result_name)
        else:
            self.exp.load(outside_result_name)
            self.exp.load(inside_result_name)

        outside_hist_name = self.compute_hist(name=outside_result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        inside_hist_name = self.compute_hist(name=inside_result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        title = 'Time distance between markings'
        xlabel = 'Time distance (s)'
        graph_name = 'marking_time_distance'
        self._plot_outside_and_inside_hist(
            graph_name, inside_hist_name, outside_hist_name, title, xlabel, normed=True, xscale='symlog', yscale='log')

    def compute_markings_shortest_distance(self, redo, redo_hist=False):
        outside_result_name = 'outside_markings_shortest_distance'
        inside_result_name = 'inside_markings_shortest_distance'
        result_name = 'markings_shortest_distance'

        dt = 5
        bins = list(np.arange(0, 300, dt))

        if redo:

            marking_name = 'manual_marking_intervals'
            xy_name = 'manual_marking_intervals_xy'
            self.exp.load([xy_name, marking_name, 'from_outside'])

            label = 'Distance between %s markings'
            description = 'Distance between the actual %s marking and the previous closest marking'

            outside_res = []
            inside_res = []
            res = []

            def get_distance4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]

                frames = list(df.index.get_level_values(id_frame_name))
                frames.sort()

                arr = df.loc[id_exp, :].reset_index().values
                for id_ant, frame, _ in arr:
                    print(id_exp, id_ant, frame)
                    id_ant = int(id_ant)
                    frame = int(frame)
                    from_outside = self.exp.get_value('from_outside', (id_exp, id_ant))
                    xy_other_markings = self.exp.get_df(xy_name).loc[id_exp, :, frame-3000:frame-1].values
                    xy = self.exp.get_df(xy_name).loc[id_exp, id_ant, frame].values
                    if len(xy_other_markings) != 0:
                        dist = round(np.min(Geo.distance(xy, xy_other_markings)), 3)
                        res.append((id_exp, id_ant, frame, dist))

                        if from_outside == 1:
                            outside_res.append((id_exp, id_ant, frame, dist))
                        else:
                            inside_res.append((id_exp, id_ant, frame, dist))

            self.exp.groupby(marking_name, id_exp_name, get_distance4each_group)

            self.exp.add_new1d_from_array(array=np.array(outside_res), name=outside_result_name, object_type='Events1d',
                                          category=self.category, label=label % 'outside',
                                          description=description % 'outside')

            self.exp.add_new1d_from_array(array=np.array(inside_res), name=inside_result_name, object_type='Events1d',
                                          category=self.category, label=label % 'inside',
                                          description=description % 'inside')

            self.exp.add_new1d_from_array(array=np.array(res), name=result_name, object_type='Events1d',
                                          category=self.category, label=label % '',
                                          description=description % '')

            self.exp.write(outside_result_name)
            self.exp.write(inside_result_name)
            self.exp.write(result_name)
        else:
            self.exp.load(outside_result_name)
            self.exp.load(inside_result_name)

        outside_hist_name = self.compute_hist(name=outside_result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        inside_hist_name = self.compute_hist(name=inside_result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        title = 'Distance between markings'
        xlabel = 'Distance (s)'
        graph_name = 'marking_shortest_distance'
        self._plot_outside_and_inside_hist(
            graph_name, inside_hist_name, outside_hist_name, title, xlabel, xscale='symlog', yscale='log')

    def get_manual_markings(self):
        result_name = 'manual_markings'
        add = '%s/MarkingMovie/markings.csv' % self.exp.root
        df = pd.read_csv(add, index_col=[id_exp_name, id_ant_name, id_frame_name])
        df2 = df['Line']
        df[df['Udi'] < 3]['Udi'] = 0
        df['Udi'][df['Udi'] == 3] = 1
        df['Udi'][df['Udi'] > 3] = 2
        mask = (df2 == -1)*(~df['Udi'].isna())
        df2[mask] = df[mask]['Udi']

        df[df['Ofer'] < 3]['Ofer'] = 0
        df['Ofer'][df['Ofer'] == 3] = 1
        df['Ofer'][df['Ofer'] > 3] = 2
        mask = (df2 == -1)*(~df['Ofer'].isna())
        df2[mask] = df[mask]['Udi']

        df2 = df2[df2 != 0]
        df2[:] = 1
        df2 = df2.astype(int)

        self.exp.load('mm10_x')

        self.exp.add_copy(old_name='mm10_x', new_name=result_name, category=self.category, label='Manual markings',
                                   description='If the ant is potentially marking, which are manually checked')

        df2 = df2.reindex(self.exp.get_index(result_name))
        df2[df2.isna()] = 0

        self.exp.change_df(result_name, df2.astype(int))

        self.exp.write(result_name)

    def compute_manual_marking_intervals(self, redo=False, redo_hist=False):

        result_name = 'manual_marking_intervals'

        bins = np.arange(0.1, 1e3, .1)
        if redo is True:
            name = 'manual_markings'
            self.exp.load(name)

            self.exp.change_df(name, 1-self.exp.get_df(name))
            self.exp.compute_time_intervals(name_to_intervals=name, category=self.category,
                                            result_name=result_name, label='Manual marking time intervals',
                                            description='Time intervals between two markings manually detected (s)')

            self.exp.write(result_name)

        hist_name = self.compute_hist(result_name, bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='Marking intervals', xscale='log', yscale='log', ylabel='PDF', ls='', normed=False)
        plotter.save(fig)

    def compute_outside_manual_marking_intervals(self, redo=False, redo_hist=False):
        typ = 'outside'
        result_name = '%s_manual_marking_intervals' % typ
        label = 'Manual %s marking time intervals' % typ
        description = 'Time intervals between two %s markings manually detected (s)' % typ
        bins = np.arange(0.1, 1e3, .1)

        if redo is True:
            name = 'manual_marking_intervals'
            self.exp.load([name, 'from_outside'])

            self.exp.add_copy(name, result_name, label=label, description=description)

            def do4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                id_ant = df.index.get_level_values(id_ant_name)[0]

                from_outside = self.exp.get_value('from_outside', (id_exp, id_ant))
                if from_outside == 0:
                    self.exp.get_df(result_name).loc[id_exp, id_ant, :] = np.nan

                return df

            self.exp.groupby(result_name, [id_exp_name, id_ant_name], do4each_group)
            self.exp.get_df(result_name).dropna(inplace=True)
            self.exp.write(result_name)

        hist_name = self.compute_hist(result_name, bins, redo=redo, redo_hist=redo_hist)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='Marking intervals', xscale='log', yscale='log', ylabel='PDF', ls='', normed=False)
        plotter.save(fig)

    def compute_inside_manual_marking_intervals(self, redo=False, redo_hist=False):
        typ = 'inside'
        result_name = '%s_manual_marking_intervals' % typ
        label = 'Manual %s marking time intervals' % typ
        description = 'Time intervals between two %s markings manually detected (s)' % typ
        bins = np.arange(0.1, 1e3, .1)

        if redo is True:
            name = 'manual_marking_intervals'
            self.exp.load([name, 'from_outside'])

            self.exp.add_copy(name, result_name, label=label, description=description)

            def do4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                id_ant = df.index.get_level_values(id_ant_name)[0]

                from_outside = self.exp.get_value('from_outside', (id_exp, id_ant))
                if from_outside == 1:
                    self.exp.get_df(result_name).loc[id_exp, id_ant, :] = np.nan

                return df

            self.exp.groupby(result_name, [id_exp_name, id_ant_name], do4each_group)
            self.exp.get_df(result_name).dropna(inplace=True)
            self.exp.write(result_name)

        hist_name = self.compute_hist(result_name, bins, redo=redo, redo_hist=redo_hist)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='Marking intervals', xscale='log', yscale='log', ylabel='PDF', ls='',
                               normed=False)
        plotter.save(fig)

    def compute_manual_marking_repartition(self, redo=False, redo_hist=False):
        outside_result_name = 'outside_manual_marking_repartition'
        inside_result_name = 'inside_manual_marking_repartition'

        dtheta = 0.5
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)

        if redo:
            label = '%s manual marking repartition'
            description = 'Angle between the food-exit vector and the food-marking vector from %s ants' \
                          ' (markings are detected manually)'

            marking_name = 'manual_marking_intervals'
            food_exit_angle_name = 'food_exit_angle'
            first_frame_name = 'first_attachment_time_of_outside_ant'
            self.exp.load([marking_name, food_exit_angle_name, first_frame_name, 'from_outside'])

            name_xy = 'xy'
            self.exp.load_as_2d('mm10_x', 'mm10_y', name_xy, 'x', 'y', replace=True)

            name_food_xy = 'food_xy'
            self.exp.load_as_2d('mm10_food_x', 'mm10_food_y', name_food_xy, 'x', 'y', replace=True)

            res_outside = []
            res_inside = []

            for id_exp, id_ant, frame in self.exp.get_index(marking_name):
                print(id_exp, id_ant, frame)
                first_frame = self.exp.get_value(first_frame_name, id_exp)
                if frame > first_frame and self.exp.get_index(name_food_xy).isin([(id_exp, frame)]).any():
                    xy = self.exp.get_df(name_xy).loc[(id_exp, id_ant, frame)]
                    xy_food = self.exp.get_df(name_food_xy).loc[(id_exp, frame)]
                    exit_angle = self.exp.get_value(food_exit_angle_name, (id_exp, frame))

                    ant_angle = Geo.angle_df(xy_food, xy)

                    angle = Geo.angle_distance(ant_angle, exit_angle)
                    from_outside = self.exp.get_value('from_outside', (id_exp, id_ant))
                    if from_outside == 1:
                        res_outside.append((id_exp, id_ant, frame, angle))
                    else:
                        res_inside.append((id_exp, id_ant, frame, angle))

            res_outside = np.array(res_outside)
            res_inside = np.array(res_inside)

            self.exp.add_new1d_from_array(array=res_outside, name=outside_result_name, object_type='Events1d',
                                          category=self.category,
                                          label=label % 'outside', description=description % 'outside')

            self.exp.add_new1d_from_array(array=res_inside, name=inside_result_name, object_type='Events1d',
                                          category=self.category,
                                          label=label % 'inside', description=description % 'inside')

            self.exp.write([outside_result_name, inside_result_name])
        else:
            self.exp.load([outside_result_name, inside_result_name])

        outside_hist_name = self.compute_hist(name=outside_result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        inside_hist_name = self.compute_hist(name=inside_result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        title = 'Marking repartition'
        xlabel = 'Marking-food-exit angle (rad)'
        graph_name = 'manual_marking_repartition'
        self._plot_outside_and_inside_hist(graph_name, inside_hist_name, outside_hist_name, title, xlabel, normed=False)

    def compute_manual_marking_intervals_xy(self):
        result_name = 'manual_marking_intervals_xy'

        marking_name = 'manual_marking_intervals'
        self.exp.load([marking_name])

        name_xy = 'xy'
        self.exp.load_as_2d('mm10_x', 'mm10_y', name_xy, 'x', 'y', replace=True)

        df = self.exp.get_df(name_xy)
        df = df.reindex(self.exp.get_index(marking_name))

        self.exp.add_new2d_from_df(df=df, name=result_name, object_type='Events2d',
                                   xname='marking_x', yname='marking_y',
                                   category=self.category, label='Coordinates of the markings (mm)',
                                   xlabel='X coordinates of the markings (mm)',
                                   ylabel='Y coordinates of the markings (mm)',
                                   description='Coordinates of the markings (mm)')

        self.exp.write(result_name)

    def compute_marking_distance(self):
        name = 'manual_marking_intervals_xy'
        self.exp.load(name)
        result_name = 'marking_distances'
        print(name)

        label = 'marking distances'
        description = 'Distance between two marking events'

        self.exp.add_new1d_from_df(df=self.exp.get_df(name).loc[:, :'marking_x'], name=result_name,
                                   object_type='Events1d', category=self.category, label=label, description=description)

        self.exp.get_df(result_name)[:] = np.nan

        def get_dist4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            id_ant = df.index.get_level_values(id_ant_name)[0]
            frames = df.index.get_level_values(id_frame_name)

            print(id_exp, id_ant)

            mark_xy = df.values
            marking_distance = Geo.distance(mark_xy[1:, :], mark_xy[:-1, :])
            marking_distance = np.around(marking_distance, 2)

            self.exp.get_df(result_name).loc[id_exp, id_ant, frames[:-1]] = np.c_[marking_distance]

            return df

        self.exp.groupby(name, [id_exp_name, id_ant_name], get_dist4each_group)

        self.exp.write(result_name)

    @staticmethod
    def __add_marking_distances(marking_distance_list, id_ant, id_exp, mark_xy):
        mark_frames = mark_xy[:-1, 2]
        lg = len(mark_frames)
        id_exp_array = np.full(lg, id_exp)
        id_ant_array = np.full(lg, id_ant)
        marking_distance = Geo.distance(mark_xy[1:, -2:], mark_xy[:-1, -2:])
        marking_distance = np.around(marking_distance, 3)
        marking_distance_list += list(zip(id_exp_array, id_ant_array, mark_frames, marking_distance))

        return marking_distance_list

    def compute_nb_manual_markings_evol_around_first_outside_attachment(self, redo=False):

        typ = ''
        name = 'manual_marking_intervals'

        result_name = 'nb_manual_markings_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.01
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(-1, 3., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of %s markings in a 10s period over time'
        description = 'Number of %s markings in a 10s period over time'

        self._get_nb_markings_evol(name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                   label % typ, description % typ, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Number of Markings',
                               label_suffix='s', marker='', display_legend=False)
        plotter.plot_smooth(preplot=(fig, ax), window=50, c='orange', label='mean smoothed')
        plotter.draw_vertical_line(ax)
        plotter.save(fig)

    def compute_nb_outside_manual_markings_evol_around_first_outside_attachment(self, redo=False):

        typ = 'outside'
        name = 'outside_manual_marking_intervals'

        result_name = 'nb_outside_manual_markings_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.01
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(-1, 3., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of %s markings in a 10s period over time'
        description = 'Number of %s markings in a 10s period over time'

        self._get_nb_markings_evol(name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                   label % typ, description % typ, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Number of Markings',
                               label_suffix='s', marker='', display_legend=False)
        plotter.plot_smooth(preplot=(fig, ax), window=50, c='orange', label='mean smoothed')
        plotter.draw_vertical_line(ax)
        plotter.save(fig)

    def compute_nb_inside_manual_markings_evol_around_first_outside_attachment(self, redo=False):

        typ = 'inside'
        name = 'inside_manual_marking_intervals'

        result_name = 'nb_inside_manual_markings_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.01
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(-1, 3., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of %s markings in a 10s period over time'
        description = 'Number of %s markings in a 10s period over time'

        self._get_nb_markings_evol(name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                   label % typ, description % typ, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Number of Markings',
                               label_suffix='s', marker='', display_legend=False)
        plotter.plot_smooth(preplot=(fig, ax), window=50, c='orange', label='mean smoothed')
        plotter.draw_vertical_line(ax)
        plotter.save(fig)


    def _get_nb_markings_evol(self, name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                              label, description, redo):
        if redo:

            self.exp.load(name)
            self.exp.load('food_x')
            self.change_first_frame(name, init_frame_name)
            self.change_first_frame('food_x', init_frame_name)

            last_frame_name = 'food_exit_frames'
            self.exp.load(last_frame_name)
            self.cut_last_frames_for_indexed_by_exp_ant_frame_indexed(name, last_frame_name)

            x = (end_frame_intervals + start_frame_intervals) / 2. / 100.
            y = np.zeros(len(start_frame_intervals))
            for i in range(len(start_frame_intervals)):
                frame0 = int(start_frame_intervals[i])
                frame1 = int(end_frame_intervals[i])

                df = self.exp.get_df(name).loc[pd.IndexSlice[:, :, frame0:frame1], :]
                df_food = self.exp.get_df('food_x').loc[pd.IndexSlice[:, frame0:frame1], :]
                y[i] = len(df) / len(df_food)*1000
            df = pd.DataFrame(y, index=x)
            self.exp.add_new_dataset_from_df(df=df, name=result_name, category=self.category,
                                             label=label, description=description)
            self.exp.write(result_name)
            self.exp.remove_object(name)

        else:
            self.exp.load(result_name)
