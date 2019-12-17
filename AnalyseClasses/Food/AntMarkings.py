import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_frame_name, id_ant_name

import Tools.MiscellaneousTools.Geometry as Geo
from Tools.Plotter.Plotter import Plotter


class AnalyseAntMarkings(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
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

    def compute_markings(self, redo=False):
        result_name = 'markings'
        label = 'Markings'
        description = 'Time of markings and if the ant is from outside'

        if redo is True:

            name_speed = 'speed'
            name_orientation = 'speed_phi'
            name_radius = 'food_radius'
            name_fps = 'fps'
            name_from_outside = 'from_outside'
            self.exp.load([name_speed, name_orientation, name_radius, name_fps, name_from_outside])
            name_xy = 'xy'
            self.exp.load_as_2d('mm10_x', 'mm10_y', name_xy, 'x', 'y', replace=True)

            name_food_xy = 'food_xy'
            self.exp.load_as_2d('mm10_food_x', 'mm10_food_y', name_food_xy, 'x', 'y', replace=True)

            dv_min = 2
            v_max = 5
            dtheta_max = 90 * np.pi / 180
            min_dist = 2

            res = []

            def get_makings4each_group(df: pd.DataFrame):

                id_exp = df.index.get_level_values(id_exp_name)[0]
                fps = self.exp.get_value(name_fps, id_exp)
                if len(df) > 2*fps:
                    id_ant = df.index.get_level_values(id_ant_name)[0]

                    print(id_exp, id_ant)
                    radius = self.exp.get_value(name_radius, id_exp)
                    from_outside = self.exp.get_value(name_from_outside, (id_exp, id_ant))

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
                                                    res.append((id_exp, id_ant, f1, from_outside))
                                        else:
                                            xys = self.exp.get_df(name_xy).loc[id_exp, :, f1]
                                            dist_ant_other_ants = Geo.squared_distance_df(xys, xy)
                                            dist_ant_other_ants = \
                                                dist_ant_other_ants.drop((id_exp, id_ant, f1)).min() / 10.
                                            if dist_ant_other_ants > min_dist:
                                                res.append((id_exp, id_ant, f1, from_outside))
                return df

            self.exp.groupby(name_speed, [id_exp_name, id_ant_name], get_makings4each_group)

            df_res = pd.DataFrame(res, columns=[id_exp_name, id_ant_name, id_frame_name, result_name])
            df_res.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)

            def do4each_group(df: pd.DataFrame):
                if len(df) < 5:
                    df[:] = np.nan
                return df

            df_res = df_res.groupby([id_exp_name, id_ant_name]).apply(do4each_group)
            df_res = df_res.dropna()
            df_res = df_res.astype(int)

            self.exp.add_new1d_from_df(df=df_res, name=result_name, object_type='Events1d', category=self.category,
                                       label=label, description=description)

            self.exp.write(result_name)

    def compute_markings_xy(self):
        result_name = 'marking_xy'

        marking_name = 'markings'
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

            marking_name = 'markings'
            food_exit_angle_name = 'food_exit_angle'
            first_frame_name = 'first_attachment_time_of_outside_ant'
            self.exp.load([marking_name, food_exit_angle_name, first_frame_name])

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
                    from_outside = self.exp.get_value(marking_name, (id_exp, id_ant, frame))
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
            self.exp.hist1d_evolution(name_to_hist=name, start_index_intervals=start_frame_intervals,
                                      end_index_intervals=end_frame_intervals, bins=bins,
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
            self.exp.hist1d_evolution(name_to_hist=name, start_index_intervals=start_frame_intervals,
                                      end_index_intervals=end_frame_intervals, bins=bins,
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
        plotter.plot(preplot=(fig, ax), xscale=xscale, yscale=yscale, normed=normed, label='outside', c='w')
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(inside_hist_name))
        plotter.plot(preplot=(fig, ax), xscale=xscale, yscale=yscale, normed=normed, label='inside')
        plotter.draw_legend(ax)
        plotter.display_title(ax, title=title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('PDF')
        plotter.save(fig, name=graph_name)

    def compute_markings_time_distance(self, redo=False, redo_hist=False):
        outside_result_name = 'outside_markings_time_distance'
        inside_result_name = 'inside_markings_time_distance'

        dt = 0.1
        bins = [-20]+list(np.arange(0, 20, dt))

        if redo:

            marking_name = 'markings'
            self.exp.load([marking_name, 'fps'])

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
                for id_ant, frame, from_outside in arr:
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
            graph_name, inside_hist_name, outside_hist_name, title, xlabel, normed=False, xscale='symlog', yscale='log')

    def compute_markings_distance(self, redo=False, redo_hist=False):
        outside_result_name = 'outside_markings_distance'
        inside_result_name = 'inside_markings_distance'

        dt = 5
        bins = list(np.arange(0, 300, dt))

        if redo:

            marking_name = 'markings'
            xy_name = 'marking_xy'
            self.exp.load([xy_name, marking_name])

            label = 'Distance between %s markings'
            description = 'Distance between the actual %s marking and the previous closest marking'

            outside_res = []
            inside_res = []

            def get_distance4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]

                frames = list(df.index.get_level_values(id_frame_name))
                frames.sort()

                arr = df.loc[id_exp, :].reset_index().values
                for id_ant, frame, from_outside in arr:
                    print(id_exp, id_ant, frame)
                    xy_other_markings = self.exp.get_df(xy_name).loc[id_exp, :, frame-3000:frame-1].values
                    xy = self.exp.get_df(xy_name).loc[id_exp, id_ant, frame].values
                    if len(xy_other_markings) != 0:
                        dist = np.min(Geo.distance(xy, xy_other_markings))

                        if from_outside == 1:
                            outside_res.append((id_exp, id_ant, frame, round(dist, 3)))
                        else:
                            inside_res.append((id_exp, id_ant, frame, round(dist, 3)))

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

        title = 'Distance between markings'
        xlabel = 'Distance (s)'
        graph_name = 'marking_distance'
        self._plot_outside_and_inside_hist(
            graph_name, inside_hist_name, outside_hist_name, title, xlabel, xscale='symlog', yscale='log')
