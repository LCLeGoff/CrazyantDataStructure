import numpy as np
import pandas as pd
from matplotlib.path import Path

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_frame_name
from Tools.MiscellaneousTools.Geometry import angle_df, angle, dot2d_df, distance_between_point_and_line_df, distance_df
from Tools.Plotter.Plotter import Plotter


class AnalyseFoodBase(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'FoodBase'

    def compute_mm10_food_traj(self):
        name_x = 'food_x'
        name_y = 'food_y'
        self.exp.load([name_x, name_y])
        time_window = 10

        result_name = self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_x, time_window=time_window,
                                                                category=self.category)
        self.exp.write(result_name)

        result_name = self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_y, time_window=time_window,
                                                                category=self.category)
        self.exp.write(result_name)

    def compute_mm1s_food_traj(self):
        name_x = 'food_x'
        name_y = 'food_y'
        time_window = 100

        result_name_x = 'mm1s_'+name_x
        result_name_y = 'mm1s_'+name_y

        self.exp.load([name_x, name_y])
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_x, result_name=result_name_x,
                                                  time_window=time_window, category=self.category)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_y, result_name=result_name_y,
                                                  time_window=time_window, category=self.category)

        self.exp.write([result_name_x, result_name_y])

    def compute_food_traj_length(self):
        result_name = 'food_traj_length'
        food_traj_name = 'food_x'
        self.exp.load([food_traj_name, 'fps'])
        self.exp.add_new1d_empty(name=result_name, object_type='Characteristics1d', category=self.category,
                                 label='Food trajectory length (s)',
                                 description='Length of the trajectory of the food in second')
        for id_exp in self.exp.id_exp_list:
            traj = self.exp.get_df(food_traj_name).loc[id_exp, :]
            frames = traj.index.get_level_values(id_frame_name)
            length = (int(frames.max())-int(frames.min()))/float(self.exp.fps.df.loc[id_exp])
            self.exp.change_value(name=result_name, idx=id_exp, value=length)

        self.exp.write(result_name)

        bins = range(-30, 500, 60)
        hist_name = self.exp.hist1d(name_to_hist=result_name, bins=bins)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot()
        ax.grid()
        ax.set_xticks(range(0, 430, 60))
        plotter.save(fig)

    def compute_food_first_frame(self):
        result_name = 'food_first_frame'
        food_traj_name = 'food_x'
        self.exp.load([food_traj_name, 'fps'])
        self.exp.add_new1d_empty(name=result_name, object_type='Characteristics1d', category=self.category,
                                 label='First frame of food trajectory',
                                 description='First frame of the trajectory of the food')
        for id_exp in self.exp.id_exp_list:
            frame0 = self.exp.get_index(food_traj_name).get_level_values(id_frame_name)[0]
            self.exp.change_value(name=result_name, idx=id_exp, value=frame0)

        self.exp.write(result_name)

    def compute_food_exit_frames(self):
        result_name = 'food_exit_frames'
        name_x = 'food_x'
        name_y = 'food_y'
        self.exp.load([name_x, name_y])
        self.exp.add_new1d_empty(name=result_name, object_type='Characteristics1d', category=self.category,
                                 label='Food trajectory length (s)',
                                 description='Length of the trajectory of the food in second')

        self.__compute_exit_pts()

        for id_exp in self.exp.id_exp_list:
            exit_path = Path([self.exp.exit1.df.loc[id_exp], self.exp.exit2.df.loc[id_exp],
                              self.exp.exit3.df.loc[id_exp], self.exp.exit4.df.loc[id_exp]])

            df_x = self.exp.get_df(name_x).loc[id_exp, :]
            df_y = self.exp.get_df(name_y).loc[id_exp, :]
            xys = np.array(df_x.join(df_y))

            is_outside = exit_path.contains_points(xys)
            id_frame = np.where(is_outside == 1)[0][0]
            frame = int(df_x.index[id_frame])

            self.exp.change_value(name=result_name, idx=id_exp, value=frame)

        self.exp.get_data_object(result_name).df = self.exp.get_df(result_name).astype(int)
        self.exp.write(result_name)

    def __compute_exit_pts(self):
        self.exp.load(['entrance1', 'entrance2', 'mm2px'])
        self.exp.add_copy('entrance1', 'exit1')
        self.exp.add_copy('entrance1', 'exit2')
        self.exp.add_copy('entrance1', 'exit3')
        self.exp.add_copy('entrance1', 'exit4')
        for id_exp in self.exp.id_exp_list:
            mm2px = self.exp.get_value('mm2px', id_exp)
            xmin = min(self.exp.entrance1.df.loc[id_exp].x, self.exp.entrance2.df.loc[id_exp].x)
            ymin = min(self.exp.entrance1.df.loc[id_exp].y, self.exp.entrance2.df.loc[id_exp].y)
            ymax = max(self.exp.entrance1.df.loc[id_exp].y, self.exp.entrance2.df.loc[id_exp].y)
            xmax = max(self.exp.entrance1.df.loc[id_exp].x, self.exp.entrance2.df.loc[id_exp].x)
            dl = 50*mm2px
            self.exp.exit1.df.x.loc[id_exp] = xmin - dl
            self.exp.exit1.df.y.loc[id_exp] = ymin - dl
            self.exp.exit2.df.x.loc[id_exp] = xmin - dl
            self.exp.exit2.df.y.loc[id_exp] = ymax + dl
            self.exp.exit3.df.x.loc[id_exp] = xmax + dl
            self.exp.exit3.df.y.loc[id_exp] = ymax + dl
            self.exp.exit4.df.x.loc[id_exp] = xmax + dl
            self.exp.exit4.df.y.loc[id_exp] = ymin - dl

        self.exp.exit1.df = self.exp.exit1.df.groupby(id_exp_name).apply(self.exp.convert_xy_to_traj_system4each_group)
        self.exp.exit2.df = self.exp.exit2.df.groupby(id_exp_name).apply(self.exp.convert_xy_to_traj_system4each_group)
        self.exp.exit3.df = self.exp.exit3.df.groupby(id_exp_name).apply(self.exp.convert_xy_to_traj_system4each_group)
        self.exp.exit4.df = self.exp.exit4.df.groupby(id_exp_name).apply(self.exp.convert_xy_to_traj_system4each_group)

    def compute_norm_time2frame(self, redo=False, redo_hist=False):
        result_name = 'norm_time2frame'

        label = 'Unit conversion normalized time to frame'
        description = 'Unit conversion from normalized time to frame'

        hist_label = 'seconds for 0.1 unit of normalized time'
        hist_description = 'Number of seconds corresponding of 0.1 unit of normalized time'

        bins = range(0, 40, 4)

        if redo:
            first_frame_name = 'food_first_frame'
            last_frame_name = 'food_exit_frames'

            self.exp.load([first_frame_name, last_frame_name])

            self.exp.add_new1d_empty(name=result_name, object_type='Characteristics1d', category=self.category,
                                     label=label, description=description)

            for id_exp in self.exp.id_exp_list:
                first_frame = self.exp.get_value(first_frame_name, id_exp)
                last_frame = self.exp.get_value(last_frame_name, id_exp)

                frame2norm_time = last_frame-first_frame
                self.exp.change_value(result_name, id_exp, frame2norm_time)

            self.exp.write(result_name)

        self.exp.operation(result_name, lambda x: x/1000.)
        hist_name = self.compute_hist(name=result_name, bins=bins, hist_label=hist_label,
                                      hist_description=hist_description, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='time (s)', ylabel='Occurrences')
        plotter.save(fig)

    def compute_food_speed(self, redo=False, redo_hist=False):
        name = 'food_speed'
        hist_name = name + '_hist'
        bins = np.arange(0, 200, 0.25)
        hist_label = 'Distribution of the food speed (mm/s)'
        hist_description = 'Distribution of the instantaneous speed of the food trajectory (mm/s)'

        if redo:
            name_x = 'mm10_food_x'
            name_y = 'mm10_food_y'
            self.exp.load([name_x, name_y, 'fps'])

            self.exp.add_copy1d(
                name_to_copy=name_x, copy_name=name, category=self.category, label='Food speed',
                description='Instantaneous speed of the food'
            )

            for id_exp in self.exp.id_exp_list:
                dx = np.array(self.exp.get_df(name_x).loc[id_exp, :])
                dx1 = dx[1, :]
                dx2 = dx[-2, :]
                dx[1:-1, :] = (dx[2:, :] - dx[:-2, :]) / 2.
                dx[0, :] = dx1 - dx[0, :]
                dx[-1, :] = dx[-1, :] - dx2

                dy = np.array(self.exp.get_df(name_y).loc[id_exp, :])
                dy1 = dy[1, :]
                dy2 = dy[-2, :]
                dy[1:-1, :] = (dy[2:, :] - dy[:-2, :]) / 2.
                dy[0, :] = dy1 - dy[0, :]
                dy[-1, :] = dy[-1, :] - dy2

                dt = np.array(self.exp.get_df(name_x).loc[id_exp, :].index.get_level_values(id_frame_name), dtype=float)
                dt.sort()
                dt[1:-1] = dt[2:] - dt[:-2]
                dt[0] = 1
                dt[-1] = 1
                dx[dt > 2] = np.nan
                dy[dt > 2] = np.nan

                food_speed = np.around(np.sqrt(dx ** 2 + dy ** 2) * self.exp.fps.df.loc[id_exp].fps, 3)
                self.exp.food_speed.df.loc[id_exp, :] = food_speed

            self.exp.write(name)

        self.compute_hist(name=name, bins=bins, hist_name=hist_name, hist_label=hist_label,
                          hist_description=hist_description, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel=r'$v$ (mm/s)', ylabel='PDF',
                               normed=True, label_suffix='s')
        ax.set_xlim((0, 20))
        plotter.save(fig)

    def compute_food_speed_evol(self, redo=False):
        name = 'food_speed'
        result_name = name + '_hist_evol'

        bins = np.arange(0, 200, 1)
        frame_intervals = np.arange(0, 5., 0.5) * 60 * 100

        if redo:
            self.exp.load(name)
            self.exp.operation(name, lambda x: np.abs(x))
            self.exp.hist1d_time_evolution(name_to_hist=name, frame_intervals=frame_intervals, bins=bins,
                                           result_name=result_name, category=self.category,
                                           label='Food speed distribution over time (rad)',
                                           description='Histogram of the instantaneous speed '
                                                       'of the food trajectory over time (rad)')
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(yscale='log', xlabel=r'$v (mm/s)$', ylabel='PDF',
                               normed=True, label_suffix='s', marker='', lw=1)
        ax.set_xlim((0, 80))
        plotter.save(fig)

    def compute_food_phi(self, redo=False, redo_hist=False):
        result_name = 'food_phi'
        hist_name = result_name+'_hist'
        print(result_name)
        name_x = 'mm10_food_x'
        name_y = 'mm10_food_y'
        self.exp.load([name_x, name_y])

        dtheta = np.pi/10.
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)
        hist_label = 'Histogram of food phi'
        hist_description = 'Histogram of the angular coordinate of the food trajectory (in the food system)'

        if redo is True:
            self.exp.add_2d_from_1ds(name1=name_x, name2=name_y, result_name='food_xy', replace=True)

            self.exp.add_copy1d(name_to_copy=name_x, copy_name=result_name, category=self.category,
                                label='food trajectory phi', description='angular coordinate of the food trajectory'
                                                                         ' (in the food system)')

            phi = np.around(angle([1, 0], self.exp.food_xy.get_array()), 3)
            self.exp.get_data_object(result_name).replace_values(phi)

            self.exp.write(result_name)

        self.compute_hist(name=result_name, bins=bins, hist_name=hist_name,
                          hist_label=hist_label, hist_description=hist_description, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel=r'$\varphi$', ylabel='PDF', normed=True)
        plotter.save(fig)

    def compute_food_exit_angle(self):
        result_name = 'food_exit_angle'
        print(result_name)

        food_x_name = 'mm10_food_x'
        food_y_name = 'mm10_food_y'
        food_name = 'food_xy'
        exit_name1 = 'exit1'
        exit_name2 = 'exit2'

        self.exp.load([food_x_name, food_y_name, exit_name1, exit_name2])

        self.exp.add_2d_from_1ds(name1=food_x_name, name2=food_y_name, result_name=food_name,
                                 xname='x', yname='y', replace=True)

        self.exp.add_copy(old_name=food_x_name, new_name=result_name, category=self.category, label='Food exit angle',
                          description='Angular coordinate of the vector formed by the food position'
                                      ' and the closest point of the exit segment')

        def compute_angle4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            print(id_exp)

            exit1 = self.exp.get_data_object('exit1').df.loc[id_exp, :]
            exit2 = self.exp.get_data_object('exit2').df.loc[id_exp, :]

            line_vector = exit2-exit1
            food = self.exp.get_df(food_name).loc[pd.IndexSlice[id_exp, :], :]

            dot1 = dot2d_df(-line_vector, exit1 - food)
            dot2 = dot2d_df(line_vector, exit2 - food)

            df.loc[(dot1 > 0) & (dot2 > 0), :] = 0
            df.loc[dot1 < 0, :] = angle_df(exit1-food).loc[dot1 < 0, :]
            df.loc[dot2 < 0, :] = angle_df(exit2-food).loc[dot2 < 0, :]

            return df

        df2 = self.exp.get_df(result_name).groupby(id_exp_name).apply(compute_angle4each_group)
        self.exp.get_data_object(result_name).df = df2

        self.exp.write(result_name)

    def compute_mm1s_food_exit_angle(self):
        name = 'food_exit_angle'
        time_window = 100

        self.exp.load(name)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name, time_window=time_window,
                                                  result_name='mm1s_' + name, category=self.category)

        self.exp.write('mm1s_' + name)

    def compute_mm10s_food_exit_angle(self):
        name = 'food_exit_angle'
        time_window = 1000

        self.exp.load(name)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name, time_window=time_window,
                                                  result_name='mm10s_' + name, category=self.category)

        self.exp.write('mm10s_' + name)

    def compute_mm30s_food_exit_angle(self):
        name = 'food_exit_angle'
        time_window = 3000

        self.exp.load(name)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name, time_window=time_window,
                                                  result_name='mm30s_' + name, category=self.category)

        self.exp.write('mm30s_' + name)

    def compute_mm60s_food_exit_angle(self):
        name = 'food_exit_angle'
        time_window = 6000

        self.exp.load(name)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name, time_window=time_window,
                                                  result_name='mm60s_' + name, category=self.category)

        self.exp.write('mm60s_' + name)

    def compute_food_exit_distance(self, redo=False, redo_hist=False):
        result_name = 'food_exit_distance'
        print(result_name)

        bins = np.arange(0, 500, 5.)

        if redo:
            food_x_name = 'mm10_food_x'
            food_y_name = 'mm10_food_y'
            food_name = 'food_xy'
            exit_name1 = 'exit1'
            exit_name2 = 'exit2'

            self.exp.load([food_x_name, food_y_name, exit_name1, exit_name2])

            self.exp.add_2d_from_1ds(name1=food_x_name, name2=food_y_name, result_name=food_name, xname='x', yname='y',
                                     replace=True)

            self.exp.add_copy(old_name=food_x_name, new_name=result_name,
                              category=self.category, label='Food exit distance (mm)',
                              description='Shortest distance between the food and the exit segment (mm)')

            def compute_angle4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                print(id_exp)

                exit1 = self.exp.get_data_object('exit1').df.loc[id_exp, :]
                exit2 = self.exp.get_data_object('exit2').df.loc[id_exp, :]

                line_vector = exit2 - exit1
                food = self.exp.get_data_object(food_name).df.loc[pd.IndexSlice[id_exp, :], :]

                dot1 = dot2d_df(-line_vector, exit1 - food)
                dot2 = dot2d_df(line_vector, exit2 - food)

                dist = distance_between_point_and_line_df(food, [exit1, exit2]).loc[(dot1 > 0) & (dot2 > 0), :]
                df.loc[(dot1 > 0) & (dot2 > 0), :] = dist

                df.loc[dot1 < 0, :] = distance_df(food, exit1).loc[dot1 < 0, :]
                df.loc[dot2 < 0, :] = distance_df(food, exit1).loc[dot2 < 0, :]

                return df.round(3)

            df2 = self.exp.get_df(result_name).groupby(id_exp_name).apply(compute_angle4each_group)
            self.exp.get_data_object(result_name).df = df2

            self.exp.write(result_name)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='distance (mm)', ylabel='PDF', normed=True)
        plotter.save(fig)

    def compute_food_exit_distance_evol(self, redo=False):
        name = 'food_exit_distance'
        result_name = name+'_hist_evol'

        bins = np.arange(0, 500, 25.)
        frame_intervals = np.arange(0, 4.5, 0.5)*60*100

        if redo:
            self.exp.load(name)
            self.exp.hist1d_time_evolution(name_to_hist=name, frame_intervals=frame_intervals, bins=bins,
                                           result_name=result_name, category=self.category)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='distance (mm)', ylabel='PDF', normed=True, label_suffix='s')
        plotter.save(fig)
