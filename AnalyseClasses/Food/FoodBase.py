import numpy as np
import pandas as pd
from matplotlib.path import Path

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_frame_name, id_ant_name
from Tools.MiscellaneousTools.Geometry import angle_df, angle, dot2d_df, distance_between_point_and_line_df, \
    distance_df, get_line_df, rotation
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

        result_name = self.exp.rolling_mean(
            name_to_average=name_x, window=time_window, category=self.category, is_angle=False)
        self.exp.write(result_name)

        result_name = self.exp.rolling_mean(
            name_to_average=name_y, window=time_window, category=self.category, is_angle=False)
        self.exp.write(result_name)

    def compute_mm1s_food_traj(self):
        name_x = 'food_x'
        name_y = 'food_y'
        time_window = 100

        result_name_x = 'mm1s_'+name_x
        result_name_y = 'mm1s_'+name_y

        self.exp.load([name_x, name_y])
        self.exp.rolling_mean(name_to_average=name_x, result_name=result_name_x,
                              window=time_window, category=self.category, is_angle=False)
        self.exp.rolling_mean(name_to_average=name_y, result_name=result_name_y,
                              window=time_window, category=self.category, is_angle=False)

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

        bins = range(0, 500, 10)
        hist_name = self.exp.hist1d(name_to_hist=result_name, bins=bins)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot()
        ax.grid()
        ax.set_xticks(range(0, 430, 60))
        plotter.save(fig)

    def compute_food_first_frame(self):
        result_name = 'food_first_frame'
        food_traj_name = 'food_x'
        self.exp.load([food_traj_name])
        self.exp.add_new1d_empty(name=result_name, object_type='Characteristics1d', category=self.category,
                                 label='First frame of food trajectory',
                                 description='First frame of the trajectory of the food')

        def get_first_frame4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            frame0 = df.index.get_level_values(id_frame_name)[0]
            self.exp.change_value(name=result_name, idx=id_exp, value=frame0)

        self.exp.groupby(food_traj_name, id_exp_name, get_first_frame4each_group)

        self.exp.change_df(result_name, self.exp.get_df(result_name).astype(int))
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
            self.exp.add_copy1d(
                name_to_copy=name_x, copy_name=name + '_x', category=self.category, label='X speed',
                description='X coordinate of the instantaneous speed of the food'
            )
            self.exp.add_copy1d(
                name_to_copy=name_x, copy_name=name + '_y', category=self.category, label='Y speed',
                description='Y coordinate of the instantaneous speed of the food'
            )

            for id_exp in self.exp.id_exp_list:
                dx = np.array(self.exp.get_df(name_x).loc[id_exp, :])
                dx1 = dx[1, :].copy()
                dx2 = dx[-2, :].copy()
                dx[1:-1, :] = (dx[2:, :] - dx[:-2, :]) / 2.
                dx[0, :] = dx1 - dx[0, :]
                dx[-1, :] = dx[-1, :] - dx2

                dy = np.array(self.exp.get_df(name_y).loc[id_exp, :])
                dy1 = dy[1, :].copy()
                dy2 = dy[-2, :].copy()
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

                fps = self.exp.get_value('fps', id_exp)
                self.exp.get_df(name).loc[id_exp, :] = np.around(np.sqrt(dx ** 2 + dy ** 2) * fps, 6)
                self.exp.get_df(name + '_x').loc[id_exp, :] = np.c_[np.around(dx * fps, 3)]
                self.exp.get_df(name + '_y').loc[id_exp, :] = np.c_[np.around(dy * fps, 3)]

            self.exp.write([name, name + '_x', name + '_y'])

        self.compute_hist(name=name, bins=bins, hist_name=hist_name, hist_label=hist_label,
                          hist_description=hist_description, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel=r'$v$ (mm/s)', ylabel='PDF',
                               normed=True, label_suffix='s')
        ax.set_xlim((0, 20))
        plotter.save(fig)

    def compute_mm10_food_speed(self):
        name = 'food_speed'
        name_x = 'food_speed_x'
        name_y = 'food_speed_y'
        names = [name, name_x, name_y]

        self.exp.load(names)
        time_window = 10

        for n in names:
            result_name = self.exp.rolling_mean(
                name_to_average=n, window=time_window, category=self.category, is_angle=False)
            self.exp.write(result_name)

    def compute_mm1s_food_speed(self):
        name = 'food_speed'
        name_x = 'food_speed_x'
        name_y = 'food_speed_y'
        names = [name, name_x, name_y]

        self.exp.load(names)
        time_window = 100

        for n in names:
            result_name = self.exp.rolling_mean(
                name_to_average=n, window=time_window, result_name='mm1s_'+n, category=self.category, is_angle=False)
            self.exp.write(result_name)

    def compute_food_speed_evol(self, redo=False):
        name = 'food_speed'
        result_name = name + '_hist_evol'

        bins = np.arange(0, 25, 1)
        dx = 0.5
        start_frame_intervals = np.arange(0, 5., dx)*60*100
        end_frame_intervals = start_frame_intervals+dx*60*100

        if redo:
            self.exp.load(name)
            self.exp.operation(name, lambda x: np.abs(x))
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                      end_frame_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category,
                                      label='Food speed distribution over time (rad)',
                                      description='Histogram of the instantaneous speed of the food trajectory over '
                                                  'time (rad)')
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$v (mm/s)$', ylabel='PDF',
                               normed=True, label_suffix='s')
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

            phi = np.around(angle(self.exp.food_xy.get_array()), 6)
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
        self.exp.get_data_object(result_name).df = np.around(df2, 6)

        self.exp.write(result_name)
        self.exp.remove_object(result_name)

    def compute_mm10_food_exit_angle(self):
        name = 'food_exit_angle'
        time_window = 10

        self.exp.load(name)
        self.exp.rolling_mean(name_to_average=name, window=time_window,
                              result_name='mm10_' + name, category=self.category, is_angle=True)

        self.exp.write('mm10_' + name)

    def compute_mm1s_food_exit_angle(self):
        name = 'food_exit_angle'
        time_window = 100

        self.exp.load(name)
        self.exp.rolling_mean(name_to_average=name, window=time_window,
                              result_name='mm1s_' + name, category=self.category, is_angle=True)

        self.exp.write('mm1s_' + name)

    def compute_mm10s_food_exit_angle(self):
        name = 'food_exit_angle'
        time_window = 1000

        self.exp.load(name)
        self.exp.rolling_mean(name_to_average=name, window=time_window,
                              result_name='mm10s_' + name, category=self.category, is_angle=True)

        self.exp.write('mm10s_' + name)

    def compute_mm30s_food_exit_angle(self):
        name = 'food_exit_angle'
        time_window = 3000

        self.exp.load(name)
        self.exp.rolling_mean(name_to_average=name, window=time_window,
                              result_name='mm30s_' + name, category=self.category, is_angle=True)

        self.exp.write('mm30s_' + name)

    def compute_mm60s_food_exit_angle(self):
        name = 'food_exit_angle'
        time_window = 6000

        self.exp.load(name)
        self.exp.rolling_mean(name_to_average=name, window=time_window,
                              result_name='mm60s_' + name, category=self.category, is_angle=True)

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
        dx = 0.5
        start_frame_intervals = np.arange(0, 4.5, dx)*60*100
        end_frame_intervals = start_frame_intervals+dx*60*100

        if redo:
            self.exp.load(name)
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                      end_frame_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='distance (mm)', ylabel='PDF', normed=True, label_suffix='s')
        plotter.save(fig)

    def compute_food_border_distance(self, redo=False):
        result_name = 'food_border_distance'
        print(result_name)

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
                              category=self.category, label='Food border distance (mm)',
                              description='Shortest distance between the food and the setup border (mm)')

            length = 420
            width = 297

            def compute_angle4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                print(id_exp)

                exit1 = self.exp.get_data_object('exit1').df.loc[id_exp, :]
                exit2 = self.exp.get_data_object('exit2').df.loc[id_exp, :]

                if exit1.y > exit2.y:
                    exit_temp = exit1.copy()
                    exit1 = exit2.copy()
                    exit2 = exit_temp

                a, b = get_line_df(exit1, exit2)
                theta = np.arctan(a)

                d = (width/2.-exit2.y)*np.sin(theta)
                y1 = exit2.y+d
                x1 = (y1-b)/a

                d = (width/2.+exit1.y)*np.sin(theta)
                y2 = exit1.y-d
                x2 = (y2-b)/a

                if a > 0:
                    theta_rotation = -theta + np.pi / 2.
                else:
                    theta_rotation = theta - np.pi / 2.

                x4, y4 = rotation(np.array([-length, 0]) + np.array([x1, y1]), theta_rotation, np.array([x1, y1]))
                x3, y3 = rotation(np.array([-length, 0]) + np.array([x2, y2]), theta_rotation, np.array([x2, y2]))

                line1 = [pd.Series({'x': x1, 'y': y1}), pd.Series({'x': x2, 'y': y2})]
                line2 = [pd.Series({'x': x2, 'y': y2}), pd.Series({'x': x3, 'y': y3})]
                line3 = [pd.Series({'x': x3, 'y': y3}), pd.Series({'x': x4, 'y': y4})]
                line4 = [pd.Series({'x': x1, 'y': y1}), pd.Series({'x': x4, 'y': y4})]

                food = self.exp.get_data_object(food_name).df.loc[pd.IndexSlice[id_exp, :], :]
                df2 = pd.DataFrame(index=df.index)
                df2['line1'] = distance_between_point_and_line_df(food, line1)
                df2['line2'] = distance_between_point_and_line_df(food, line2)
                df2['line3'] = distance_between_point_and_line_df(food, line3)
                df2['line4'] = distance_between_point_and_line_df(food, line4)

                df[:] = np.c_[df2.min(axis=1)]

                return df.round(3)

            df_res = self.exp.get_df(result_name).groupby(id_exp_name).apply(compute_angle4each_group)
            self.exp.get_data_object(result_name).df = df_res

            self.exp.write(result_name)

    def compute_food_r(self, redo=False):
        res_name = 'food_r'
        label = 'Radial coordinate of the food (mm)'
        description = 'Radial coordinate of the food (mm)'

        if redo:
            name_x = 'food_x'
            name_y = 'food_y'
            self.exp.load_as_2d(name_x, name_y, 'food_xy', replace=True)

            d = distance_df(self.exp.get_df('food_xy'))

            self.exp.add_new1d_from_df(df=d, name=res_name, object_type='CharacteristicTimeSeries1d',
                                       category=self.category, label=label, description=description)
            self.exp.write(res_name)

    def compute_food_r_mean_evol(self, redo=False):

        name = 'food_r'
        result_name = name + '_mean_evol'
        init_frame_name = 'food_first_frame'

        dx = 1/12.
        start_frame_intervals = np.arange(0, 4., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        label = 'Mean of the food radial coordinate over time'
        description = 'Mean of the radial coordinate of the food over time'

        if redo:
            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            self.exp.mean_evolution(name_to_var=name, start_index_intervals=start_frame_intervals,
                                    end_index_intervals=end_frame_intervals,
                                    category=self.category, result_name=result_name,
                                    label=label, description=description)

            self.exp.write(result_name)
            self.exp.remove_object(name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Mean', label_suffix='s', label='Mean')
        plotter.plot_fit(preplot=(fig, ax), typ='linear')
        plotter.save(fig)

    def compute_food_r_mean_evol_around_first_attachment(self, redo=False):

        name = 'food_r'
        result_name = name + '_mean_evol_first_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 1/12.
        start_frame_intervals = np.arange(-2, 3., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        label = 'Mean of the food radial coordinate over time'
        description = 'Mean of the radial coordinate of the food over time'

        if redo:

            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            self.exp.mean_evolution(name_to_var=name, start_index_intervals=start_frame_intervals,
                                    end_index_intervals=end_frame_intervals,
                                    category=self.category, result_name=result_name,
                                    label=label, description=description)

            self.exp.write(result_name)
            self.exp.remove_object(name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Mean', label_suffix='s', label='Mean')
        plotter.plot_fit(preplot=(fig, ax), typ='linear')
        plotter.draw_vertical_line(ax)
        plotter.save(fig)

    def compute_food_r_var_evol(self, redo=False):

        name = 'food_r'
        result_name = name + '_var_evol'
        init_frame_name = 'food_first_frame'

        dx = 0.25
        start_frame_intervals = np.arange(0, 4., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        label = 'Variance of the food radial coordinate over time'
        description = 'Variance of the radial coordinate of the food over time'

        if redo:

            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            self.exp.variance_evolution(name_to_var=name, start_index_intervals=start_frame_intervals,
                                        end_index_intervals=end_frame_intervals,
                                        category=self.category, result_name=result_name,
                                        label=label, description=description)

            self.exp.write(result_name)
            self.exp.remove_object(name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Variance', label_suffix='s', label='Variance')
        plotter.plot_fit(preplot=(fig, ax), typ='linear')
        plotter.save(fig)

    def compute_food_r_var_evol_around_first_attachment(self, redo=False):

        name = 'food_r'
        result_name = name + '_var_evol_around_first_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.25
        start_frame_intervals = np.arange(-2, 3., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        label = 'Variance of the food radial coordinate over time'
        description = 'Variance of the radial coordinate of the food over time'

        if redo:

            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            self.exp.variance_evolution(name_to_var=name, start_index_intervals=start_frame_intervals,
                                        end_index_intervals=end_frame_intervals,
                                        category=self.category, result_name=result_name,
                                        label=label, description=description)

            self.exp.write(result_name)
            self.exp.remove_object(name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Variance', label_suffix='s', label='Variance')
        plotter.plot_fit(preplot=(fig, ax), typ='linear')
        plotter.save(fig)

    def compute_food_phi_var_evol(self, redo=False):

        name = 'food_phi'
        result_name = name + '_var_evol'
        init_frame_name = 'food_first_frame'

        dx = 0.25
        start_frame_intervals = np.arange(0, 4., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        label = 'Variance of the food angular coordinate over time'
        description = 'Variance of the angular coordinate of the food over time'

        if redo:

            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            self.exp.variance_evolution(name_to_var=name, start_index_intervals=start_frame_intervals,
                                        end_index_intervals=end_frame_intervals,
                                        category=self.category, result_name=result_name,
                                        label=label, description=description)

            self.exp.write(result_name)
            self.exp.remove_object(name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Variance', label_suffix='s', label='Variance')
        plotter.plot_fit(preplot=(fig, ax), typ='linear')
        plotter.save(fig)

    def compute_food_phi_var_evol_around_first_attachment(self, redo=False):

        name = 'food_phi'
        result_name = name + '_var_evol_around_first_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.1
        dx2 = 0.01
        start_frame_intervals = np.arange(-1, 3., dx2)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        label = 'Variance of the food angular coordinate over time'
        description = 'Variance of the angular coordinate of the food over time'

        if redo:

            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            self.exp.variance_evolution(name_to_var=name, start_index_intervals=start_frame_intervals,
                                        end_index_intervals=end_frame_intervals,
                                        category=self.category, result_name=result_name,
                                        label=label, description=description)

            self.exp.write(result_name)
            self.exp.remove_object(name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(
            xlabel='Time (s)', ylabel='Variance', label_suffix='s', label='Variance', yscale='log', marker='')
        plotter.plot_fit(preplot=(fig, ax), typ='exp', window=[90, 400])
        plotter.draw_vertical_line(ax)
        plotter.save(fig)

    def __reindexing_food_xy(self, name):
        id_exps = self.exp.get_df(name).index.get_level_values(id_exp_name)
        id_ants = self.exp.get_df(name).index.get_level_values(id_ant_name)
        frames = self.exp.get_df(name).index.get_level_values(id_frame_name)
        idxs = pd.MultiIndex.from_tuples(list(zip(id_exps, frames)), names=[id_exp_name, id_frame_name])

        df_d = self.exp.get_df('food_xy').copy()
        df_d = df_d.reindex(idxs)
        df_d[id_ant_name] = id_ants
        df_d.reset_index(inplace=True)
        df_d.columns = [id_exp_name, id_frame_name, 'x', 'y', id_ant_name]
        df_d.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)
        return df_d

    def compute_food_phi_hist_evol(self, redo=False):
        name = 'food_phi'
        result_name = name+'_hist_evol'
        init_frame_name = 'food_first_frame'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        dx = 0.25
        start_frame_intervals = np.array(np.arange(0, 3., dx)*60*100, dtype=int)
        end_frame_intervals = np.array(start_frame_intervals + dx*60*100*2, dtype=int)

        func = lambda a: np.abs(a)

        hist_label = 'Food phi distribution over time (rad)'
        hist_description = 'Histogram of the angular coordinate of the food (rad)'

        if redo:
            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            self.exp.operation(name, func)
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                      end_frame_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category,
                                      label=hist_label, description=hist_description)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\phi$ (rad)', ylabel='PDF', normed=True, label_suffix='s',
                               title='')
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def compute_food_phi_hist_evol_around_first_outside_attachment(self, redo=False):
        name = 'food_phi'
        result_name = name+'_hist_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        dx = 0.25
        start_frame_intervals = np.array(np.arange(-1, 3., dx)*60*100, dtype=int)
        end_frame_intervals = np.array(start_frame_intervals + dx*60*100*2, dtype=int)

        func = lambda a: np.abs(a)

        hist_label = 'Food phi distribution over time (rad)'
        hist_description = 'Histogram of the angular coordinate of the food (rad)'

        if redo:
            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            self.exp.operation(name, func)
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                      end_frame_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category,
                                      label=hist_label, description=hist_description)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\phi$ (rad)', ylabel='PDF', normed=True, label_suffix='s',
                               title='')
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)
