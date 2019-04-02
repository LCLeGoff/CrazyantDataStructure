import numpy as np
import pandas as pd

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.MiscellaneousTools.Geometry import angle_df, norm_angle_tab, norm_angle_tab2, angle, dot2d_df, \
    distance_between_point_and_line_df, distance_df
from Tools.Plotter.Plotter import Plotter


class AnalyseFoodBase(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)

    def compute_food_traj_length(self):
        result_name = 'food_traj_length'
        food_traj_name = 'food_x'
        self.exp.load([food_traj_name, 'fps'])
        self.exp.add_new1d_empty(name=result_name, object_type='Characteristics1d', category='FoodBase',
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

    def compute_food_phi(self, redo=False, redo_hist=False):
        result_name = 'food_phi'
        hist_name = result_name+'_hist'
        print(result_name)
        self.exp.load(['food_x', 'food_y'])

        dtheta = np.pi/10.
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)
        hist_label = 'Histogram of food phi'
        hist_description = 'Histogram of the angular coordinate of the food trajectory (in the food system)'

        if redo is True:
            self.exp.add_2d_from_1ds(name1='food_x', name2='food_y', result_name='food_xy')

            self.exp.add_copy1d(name_to_copy='food_x', copy_name=result_name, category='FoodBase',
                                label='food trajectory phi', description='angular coordinate of the food trajectory'
                                                                         ' (in the food system)')

            phi = np.around(angle([1, 0], self.exp.food_xy.get_array()), 3)
            self.exp.get_data_object(result_name).replace_values(phi)

            self.exp.write(result_name)

            self.exp.hist1d(name_to_hist=result_name, bins=bins, label=hist_label, description=hist_description)
            self.exp.write(hist_name)

        elif redo_hist is True:
            self.exp.load(result_name)
            self.exp.hist1d(name_to_hist=result_name, bins=bins, label=hist_label, description=hist_description)
            self.exp.write(hist_name)

        else:
            self.exp.load(hist_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel=r'$\varphi$', ylabel='PDF', normed=True)
        plotter.save(fig)

    def compute_food_exit_angle(self):
        result_name = 'food_exit_angle'
        print(result_name)

        food_x_name = 'food_x'
        food_y_name = 'food_y'
        food_name = 'food_xy'
        exit_name1 = 'exit1'
        exit_name2 = 'exit2'

        self.exp.load([food_x_name, food_y_name, exit_name1, exit_name2])

        self.exp.add_2d_from_1ds(name1=food_x_name, name2=food_y_name, result_name=food_name, xname='x', yname='y')

        self.exp.add_copy(old_name=food_x_name, new_name=result_name, category='FoodBase', label='Food exit angle',
                          description='Angular coordinate of the vector formed by the food position'
                                      ' and the closest point of the exit segment')

        def compute_angle4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            print(id_exp)

            exit1 = self.exp.get_data_object('exit1').df.loc[id_exp, :]
            exit2 = self.exp.get_data_object('exit2').df.loc[id_exp, :]

            line_vector = exit2-exit1
            food = self.exp.get_data_object(food_name).df.loc[pd.IndexSlice[id_exp, :], :]

            dot1 = dot2d_df(-line_vector, exit1 - food)
            dot2 = dot2d_df(line_vector, exit2 - food)

            df.loc[(dot1 > 0) & (dot2 > 0), :] = 0
            df.loc[dot1 < 0, :] = angle_df(exit1-food).loc[dot1 < 0, :]
            df.loc[dot2 < 0, :] = angle_df(exit2-food).loc[dot2 < 0, :]

            return df

        df = self.exp.get_df(result_name).groupby(id_exp_name).apply(compute_angle4each_group)
        self.exp.get_data_object(result_name).df = df

        self.exp.write(result_name)

    def compute_food_exit_distance(self, redo=False, redo_hist=False):
        result_name = 'food_exit_distance'
        print(result_name)

        bins = np.arange(0, 500, 5.)

        if redo:
            food_x_name = 'food_x'
            food_y_name = 'food_y'
            food_name = 'food_xy'
            exit_name1 = 'exit1'
            exit_name2 = 'exit2'

            self.exp.load([food_x_name, food_y_name, exit_name1, exit_name2])

            self.exp.add_2d_from_1ds(name1=food_x_name, name2=food_y_name, result_name=food_name, xname='x', yname='y',
                                     replace=True)

            self.exp.add_copy(old_name=food_x_name, new_name=result_name,
                              category='FoodBase', label='Food exit distance (mm)',
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

            df = self.exp.get_df(result_name).groupby(id_exp_name).apply(compute_angle4each_group)
            self.exp.get_data_object(result_name).df = df

            self.exp.write(result_name)

        hist_name = self.compute_hist(result_name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='distance (mm)', ylabel='PDF', normed=True)
        plotter.save(fig)

    def compute_food_exit_distance_evol(self, redo=False):
        name = 'food_exit_distance'
        result_name = name+'_hist_evol'

        bins = np.arange(0, 500, 10.)
        frame_intervals = np.arange(0, 5., 0.5)*60*100

        if redo:
            self.exp.load(name)
            self.exp.hist1d_time_evolution(name_to_hist=name, frame_intervals=frame_intervals, bins=bins,
                                           result_name=result_name, category='FoodBase')
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='distance (mm)', ylabel='PDF', normed=True, label_suffix='s')
        plotter.save(fig)

    def compute_food_velocity_phi(self, redo=False, redo_hist=False):
        name_x = 'food_x'
        name_y = 'food_y'
        category = 'FoodBase'

        result_velocity_name = 'food_velocity_phi'
        hist_name = result_velocity_name+'_hist'
        dtheta = np.pi/25.
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)

        hist_label = 'Histogram of food velocity phi (rad)'
        hist_description = 'Histogram of the angular coordinate of the food velocity (rad, in the food system)'
        if redo:
            self.exp.load([name_x, name_y])

            self.exp.add_copy1d(
                name_to_copy=name_x, copy_name=result_velocity_name, category=category, label='Food velocity phi (rad)',
                description='Angular coordinate of the food velocity (rad, in the food system)'
            )

            for id_exp in self.exp.characteristic_timeseries_exp_frame_index:

                dx = np.array(self.exp.food_x.df.loc[id_exp, :]).ravel()
                dx1 = dx[1].copy()
                dx2 = dx[-2].copy()
                dx[1:-1] = (dx[2:] - dx[:-2]) / 2.
                dx[0] = dx1 - dx[0]
                dx[-1] = dx[-1] - dx2

                dy = np.array(self.exp.food_y.df.loc[id_exp, :]).ravel()
                dy1 = dy[1].copy()
                dy2 = dy[-2].copy()
                dy[1:-1] = (dy[2:] - dy[:-2]) / 2.
                dy[0] = dy1 - dy[0]
                dy[-1] = dy[-1] - dy2

                dvel_phi = angle(np.array(list(zip(dx, dy))))
                self.exp.get_df(result_velocity_name).loc[id_exp, :] = np.around(dvel_phi, 3)

            self.exp.write(result_velocity_name)

        self.compute_hist(hist_name=hist_name, result_name=result_velocity_name, bins=bins,
                          hist_label=hist_label, hist_description=hist_description, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel=r'$\varphi$', ylabel='PDF', normed=True)
        plotter.save(fig)

    def compute_food_velocity_phi_evol(self, redo=False):
        name = 'food_velocity_phi'
        result_name = name+'_hist_evol'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)
        frame_intervals = np.arange(0, 5., 0.5)*60*100

        if redo:
            self.exp.load(name)
            self.exp.operation(name, lambda x: np.abs(x))
            self.exp.hist1d_time_evolution(name_to_hist=name, frame_intervals=frame_intervals, bins=bins,
                                           result_name=result_name, category='FoodBase',
                                           label='Food velocity phi distribution over time (rad)',
                                           description='Histogram of the absolute value of the angular coordinate'
                                                       ' of the velocity of the food trajectory over time (rad)')
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\varphi$', ylabel='PDF', normed=True, label_suffix='s')
        plotter.save(fig)

    def compute_food_direction_error(self, redo=False, redo_hist=False):
        food_exit_angle_name = 'food_exit_angle'
        food_phi_name = 'food_velocity_phi'
        result_name = 'food_direction_error'
        category = 'FoodBase'

        hist_name = result_name+'_hist'
        dtheta = np.pi/25.
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)

        result_label = 'Food direction error (rad)'
        result_description = 'Angle between the food velocity and the food-exit vector,' \
                             'which gives in radian how much the food is not going in the good direction'
        if redo:
            self.exp.load([food_exit_angle_name, food_phi_name])

            tab = self.exp.get_df(food_exit_angle_name)[food_exit_angle_name] \
                - self.exp.get_df(food_phi_name)[food_phi_name]

            self.exp.add_copy1d(name_to_copy=food_phi_name, copy_name=result_name, category=category,
                                label=result_label, description=result_description)

            self.exp.change_values(result_name, np.around(tab, 3))

            self.exp.write(result_name)

        self.compute_hist(hist_name=hist_name, result_name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel=r'$\varphi_{error}$', ylabel='PDF', normed=True)
        plotter.save(fig)

    def compute_food_direction_error_evol(self, redo=False):
        name = 'food_direction_error'
        result_name = name+'_hist_evol'
        category = 'FoodBase'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)
        frame_intervals = np.arange(0, 5., .5)*60*100
        func = lambda x: np.abs(x)

        hist_label = 'Food direction error distribution over time (rad)'
        hist_description = 'Histogram of the angle between the food velocity and the food-exit vector,' \
                           'which gives in radian how much the food is not going in the good direction (rad)'

        if redo:
            self.exp.load(name)
            self.exp.operation(name, func)
            self.exp.hist1d_time_evolution(name_to_hist=name, frame_intervals=frame_intervals, bins=bins,
                                           result_name=result_name, category=category,
                                           label=hist_label, description=hist_description)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\varphi$', ylabel='PDF', normed=True, label_suffix='s')
        plotter.save(fig)

    def compute_mm1s_food_direction_error(self):
        name = 'food_direction_error'
        category = 'FoodBase'
        time_window = 100

        self.exp.load(name)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name, time_window=time_window,
                                                      result_name='mm1s_'+name, category=category)

        self.exp.write('mm1s_'+name)

    def compute_distance2food(self):
        name = 'distance2food'
        print(name)
        self.exp.load(['food_x', 'food_y', 'x', 'y'])

        id_exps = self.exp.x.df.index.get_level_values(id_exp_name)
        id_ants = self.exp.x.df.index.get_level_values(id_ant_name)
        frames = self.exp.x.df.index.get_level_values(id_frame_name)
        idxs = pd.MultiIndex.from_tuples(list(zip(id_exps, frames)), names=[id_exp_name, id_frame_name])

        self.exp.add_2d_from_1ds(
            name1='food_x', name2='food_y', result_name='food_xy',
            xname='x', yname='y'
        )

        df_d = self.__reindexing_food_xy(id_ants, idxs)
        df_d = self.__compute_distance_from_food(df_d)

        self.exp.add_new1d_from_df(
            df=df_d, name=name, object_type='TimeSeries1d',
            category='FoodBase', label='Food to distance', description='Distance between the food and the ants'
        )
        self.exp.write(name)

    def __reindexing_food_xy(self, id_ants, idxs):
        df_d = self.exp.food_xy.df.copy()
        df_d = df_d.reindex(idxs)
        df_d[id_ant_name] = id_ants
        df_d.reset_index(inplace=True)
        df_d.columns = [id_exp_name, id_frame_name, 'x', 'y', id_ant_name]
        df_d.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)
        return df_d

    def __compute_distance_from_food(self, df_f):
        df_d = np.around(np.sqrt((df_f.x - self.exp.x.df.x) ** 2 + (df_f.y - self.exp.y.df.y) ** 2), 3)
        df_d = pd.DataFrame(df_d)
        return df_d

    def compute_mm10_distance2food(self):
        name = 'distance2food'
        category = 'Distance2foodMM'
        time_window = 10

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=category
        )

        self.exp.write(result_name)

    def compute_mm20_distance2food(self):
        name = 'distance2food'
        category = 'Distance2foodMM'
        time_window = 20

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=category
        )

        self.exp.write(result_name)

    def compute_speed_food(self, redo=False, redo_hist=False):
        name = 'food_speed'
        hist_name = name+'_hist'
        bins = np.arange(0, 200, 1)
        hist_label = 'Distribution of the food speed (mm/s)'
        hist_description = 'Distribution of the instantaneous speed of the food trajectory (mm/s)'

        if redo:
            self.exp.load(['food_x', 'food_y', 'fps'])

            self.exp.add_copy1d(
                name_to_copy='food_x', copy_name=name, category='FoodBase', label='Food speed',
                description='Instantaneous speed of the food'
            )

            for id_exp in self.exp.characteristic_timeseries_exp_frame_index:
                dx = np.array(self.exp.food_x.df.loc[id_exp, :])
                dx1 = dx[1, :]
                dx2 = dx[-2, :]
                dx[1:-1, :] = (dx[2:, :]-dx[:-2, :])/2.
                dx[0, :] = dx1 - dx[0, :]
                dx[-1, :] = dx[-1, :] - dx2

                dy = np.array(self.exp.food_y.df.loc[id_exp, :])
                dy1 = dy[1, :]
                dy2 = dy[-2, :]
                dy[1:-1, :] = (dy[2:, :]-dy[:-2, :])/2.
                dy[0, :] = dy1 - dy[0, :]
                dy[-1, :] = dy[-1, :] - dy2

                dt = np.array(self.exp.characteristic_timeseries_exp_frame_index[id_exp], dtype=float)
                dt.sort()
                dt[1:-1] = dt[2:]-dt[:-2]
                dt[0] = 1
                dt[-1] = 1
                dx[dt > 2] = np.nan
                dy[dt > 2] = np.nan

                food_speed = np.around(np.sqrt(dx ** 2 + dy ** 2) * self.exp.fps.df.loc[id_exp].fps, 3)
                self.exp.food_speed.df.loc[id_exp, :] = food_speed

            self.exp.write(name)

        if redo or redo_hist:
            self.exp.load(name)
            self.exp.hist1d(name_to_hist=name, result_name=hist_name,
                            bins=bins, label=hist_label, description=hist_description)
            self.exp.write(hist_name)

        else:
            self.exp.load(hist_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(yscale='log', xlabel=r'$\dot\varphi_{food} (mm/s)$', ylabel='PDF',
                               normed=True, label_suffix='s')
        plotter.save(fig)

    def compute_is_xy_next2food(self):
        name = 'is_xy_next2food'

        name_distance = 'distance2food'
        self.exp.load(name_distance)
        self.exp.add_copy1d(
            name_to_copy=name_distance, copy_name=name, category='FoodBase',
            label='Is next to food?', description='Is ants next to the food?'
        )

        neighbor_distance = 15.
        neighbor_distance2 = 5.
        self.exp.operation(name, lambda x: (x < neighbor_distance)*(x > neighbor_distance2))
        self.exp.is_xy_next2food.df = self.exp.is_xy_next2food.df.astype(int)
        self.exp.write(name)

    def compute_xy_next2food(self):
        name = 'xy_next2food'

        self.exp.load(['x', 'y', 'is_xy_next2food'])

        self.exp.add_2d_from_1ds(
            name1='x', name2='y', result_name='xy'
        )

        self.exp.filter_with_values(
            name_to_filter='xy', filter_name='is_xy_next2food', result_name=name,
            xname='x', yname='y', category='FoodBase',
            label='XY next to food', xlabel='x', ylabel='y', description='Trajectory of ant next to food'
        )

        self.exp.write(name)

    def compute_speed_xy_next2food(self):
        name = 'speed_xy_next2food'

        self.exp.load(['speed_x', 'speed_y', 'is_xy_next2food'])

        self.exp.add_2d_from_1ds(
            name1='speed_x', name2='speed_y', result_name='dxy'
        )

        self.exp.filter_with_values(
            name_to_filter='dxy', filter_name='is_xy_next2food', result_name=name,
            xname='x', yname='y', category='FoodBase',
            label='speed vector next to food', xlabel='x', ylabel='y', description='Speed vector of ants next to food'
        )

        self.exp.write(name)

    def compute_speed_next2food(self):
        name = 'speed'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='speed next to food', description='Instantaneous speed of ant next to food'
        )

        self.exp.write(res_name)

    def compute_mm10_speed_next2food(self):
        name = 'mm10_speed'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name, label='speed next to food',
            description='Moving mean (time window of 10 frames) of the instantaneous speed of ant close to the food'
        )

        self.exp.write(res_name)

    def compute_mm20_speed_next2food(self):
        name = 'mm20_speed'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name, label='speed next to food',
            description='Moving mean (time window of 20 frames) of the instantaneous speed of ant close to the food'
        )

        self.exp.write(res_name)

    def compute_distance2food_next2food(self):
        name = 'distance2food'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='Food distance next to food',
            description='Distance between the food and the ants next to the food'
        )

        self.exp.write(res_name)

    def compute_mm5_distance2food_next2food(self):
        name = 'mm5_distance2food'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            label='Food distance next to food',
            description='Moving mean (time window of 5 frames) '
                        'of the distance between the food and the ants next to the food'
        )

        self.exp.write(res_name)

    def compute_mm10_distance2food_next2food(self):
        name = 'mm10_distance2food'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            label='Food distance next to food',
            description='Moving mean (time window of 10 frames) '
                        'of the distance between the food and the ants next to the food'
        )

        self.exp.write(res_name)

    def compute_mm20_distance2food_next2food(self):
        name = 'mm20_distance2food'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            label='Food distance next to food',
            description='Moving mean (time window of 20 frames) '
                        'of the distance between the food and the ants next to the food'
        )

        self.exp.write(res_name)

    def __diff4each_group(self, df: pd.DataFrame):
        name0 = df.columns[0]
        df.dropna(inplace=True)
        id_exp = df.index.get_level_values(id_exp_name)[0]
        d = np.array(df)
        if len(d) > 1:

            d1 = d[1].copy()
            d2 = d[-2].copy()
            d[1:-1] = (d[2:] - d[:-2]) / 2.
            d[0] = d1-d[0]
            d[-1] = d[-1]-d2

            dt = np.array(df.index.get_level_values(id_frame_name), dtype=float)
            dt[1:-1] = dt[2:] - dt[:-2]
            dt[0] = 1
            dt[-1] = 1
            d[dt > 2] = np.nan

            df[name0] = d * self.exp.fps.df.loc[id_exp].fps
        else:
            df[name0] = np.nan

        return df

    def compute_distance2food_next2food_differential(self):
        name = 'distance2food_next2food'
        result_name = 'distance2food_next2food_diff'

        self.exp.load([name, 'fps'])

        self.exp.add_copy(
            old_name=name, new_name=result_name, category='FoodBase', label='Food distance differential',
            description='Differential of the distance between the food and the ants', replace=True
        )

        self.exp.get_data_object(result_name).change_values(
            self.exp.get_df(result_name).groupby([id_exp_name, id_ant_name]).apply(self.__diff4each_group))

        self.exp.write(result_name)

    def compute_mm10_distance2food_next2food_differential(self):
        name = 'mm10_distance2food_next2food'
        result_name = 'mm10_distance2food_next2food_diff'

        self.exp.load([name, 'fps'])

        self.exp.add_copy(
            old_name=name, new_name=result_name, label='Food distance differential',
            description='Differential of the distance between the food and the ants', replace=True
        )

        self.exp.get_data_object(result_name).change_values(
            self.exp.get_df(result_name).groupby([id_exp_name, id_ant_name]).apply(self.__diff4each_group))

        self.exp.write(result_name)

    def compute_mm20_distance2food_next2food_differential(self):
        name = 'mm20_distance2food_next2food'
        result_name = 'mm20_distance2food_next2food_diff'

        self.exp.load([name, 'fps'])

        self.exp.add_copy(
            old_name=name, new_name=result_name, label='Food distance differential',
            description='Differential of the distance between the food and the ants', replace=True
        )

        self.exp.get_data_object(result_name).change_values(
            self.exp.get_df(result_name).groupby([id_exp_name, id_ant_name]).apply(self.__diff4each_group))

        self.exp.write(result_name)

    def compute_orientation_next2food(self):
        name = 'orientation'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            label='orientation next to food', description='Body orientation of ant next to food'
        )

        self.exp.write(res_name)

    def compute_mm10_orientation_next2food(self):
        name = 'mm10_orientation'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            label='orientation next to food',
            description='Moving mean (time window of 10 frames) of the body orientation of ant next to food'
        )

        self.exp.write(res_name)

    def compute_mm20_orientation_next2food(self):
        name = 'mm20_orientation'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            label='orientation next to food',
            description='Moving mean (time window of 20 frames) of the body orientation of ant next to food'
        )

        self.exp.write(res_name)

    def compute_angle_body_food(self):
        name = 'angle_body_food'

        self.exp.load(['food_x', 'food_y', 'x', 'y', 'orientation'])

        self.exp.add_2d_from_1ds(
            name1='x', name2='y', result_name='xy',
            xname='x', yname='y', replace=True
        )
        id_exps = self.exp.xy.df.index.get_level_values(id_exp_name)
        id_ants = self.exp.xy.df.index.get_level_values(id_ant_name)
        frames = self.exp.xy.df.index.get_level_values(id_frame_name)
        idxs = pd.MultiIndex.from_tuples(list(zip(id_exps, frames)), names=[id_exp_name, id_frame_name])
        self.exp.add_2d_from_1ds(
            name1='food_x', name2='food_y', result_name='food_xy',
            xname='x_ant', yname='y_ant', replace=True
        )
        df_food = self.__reindexing_food_xy(id_ants, idxs)

        df_ant_vector = df_food.copy()
        df_ant_vector.x = df_food.x - self.exp.xy.df.x
        df_ant_vector.y = df_food.y - self.exp.xy.df.y
        self.exp.add_copy('orientation', 'ant_food_orientation')
        self.exp.ant_food_orientation.change_values(angle_df(df_ant_vector))

        self.exp.add_copy(
            old_name='orientation', new_name=name, category='FoodBase', label='Body theta_res to food',
            description='Angle between the ant-food vector and the body vector', replace=True
        )
        self.exp.get_data_object(name).change_values(norm_angle_tab(
            self.exp.ant_food_orientation.df.ant_food_orientation
            - self.exp.orientation.df.orientation))
        self.exp.operation(name, lambda x: np.around(norm_angle_tab2(x), 3))

        self.exp.write(name)

    def compute_mm5_angle_body_food(self):
        name = 'angle_body_food'
        category = 'Distance2foodMM'
        time_window = 5

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=category
        )

        self.exp.write(result_name)

    def compute_mm10_angle_body_food(self):
        name = 'angle_body_food'
        category = 'Distance2foodMM'
        time_window = 10

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=category
        )

        self.exp.write(result_name)

    def compute_mm20_angle_body_food(self):
        name = 'angle_body_food'
        category = 'Distance2foodMM'
        time_window = 20

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=category
        )

        self.exp.write(result_name)

    def compute_angle_body_food_next2food(self):
        name = 'angle_body_food'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='orientation next to food',
            description='Angle between the ant-food vector and the body vector for the ants close to the food'
        )

        self.exp.write(res_name)

    def compute_mm5_angle_body_food_next2food(self):
        name = 'mm5_angle_body_food'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='orientation next to food',
            description='Moving mean (time window of 5 frames)  of the angle'
                        ' between the ant-food vector and the body vector for the ants close to the food'
        )

        self.exp.write(res_name)

    def compute_mm10_angle_body_food_next2food(self):
        name = 'mm10_angle_body_food'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='orientation next to food',
            description='Moving mean (time window of 10 frames)  of the angle'
                        ' between the ant-food vector and the body vector for the ants close to the food'
        )

        self.exp.write(res_name)

    def compute_mm20_angle_body_food_next2food(self):
        name = 'mm20_angle_body_food'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='orientation next to food',
            description='Moving mean (time window of 20 frames)  of the angle'
                        ' between the ant-food vector and the body vector for the ants close to the food'
        )

        self.exp.write(res_name)
