import numpy as np
import pandas as pd

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.MiscellaneousTools.ArrayManipulation import get_entropy
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

        self.compute_hist(name=result_name, bins=bins, hist_name=hist_name,
                          hist_label=hist_label, hist_description=hist_description, redo=redo, redo_hist=redo_hist)

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
        category = 'FoodBase'
        time_window = 100

        self.exp.load(name)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name, time_window=time_window,
                                                  result_name='mm1s_' + name, category=category)

        self.exp.write('mm1s_' + name)

    def compute_mm10s_food_exit_angle(self):
        name = 'food_exit_angle'
        category = 'FoodBase'
        time_window = 1000

        self.exp.load(name)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name, time_window=time_window,
                                                  result_name='mm10s_' + name, category=category)

        self.exp.write('mm10s_' + name)

    def compute_mm30s_food_exit_angle(self):
        name = 'food_exit_angle'
        category = 'FoodBase'
        time_window = 3000

        self.exp.load(name)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name, time_window=time_window,
                                                  result_name='mm30s_' + name, category=category)

        self.exp.write('mm30s_' + name)

    def compute_mm60s_food_exit_angle(self):
        name = 'food_exit_angle'
        category = 'FoodBase'
        time_window = 6000

        self.exp.load(name)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name, time_window=time_window,
                                                  result_name='mm60s_' + name, category=category)

        self.exp.write('mm60s_' + name)

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

    def compute_food_velocity(self, redo=False, redo_hist=False):
        name_x = 'food_x'
        name_y = 'food_y'
        category = 'FoodBase'

        result_velocity_phi_name = 'food_velocity_phi'
        result_velocity_x_name = 'food_velocity_x'
        result_velocity_y_name = 'food_velocity_y'

        hist_name = result_velocity_phi_name+'_hist'
        dtheta = np.pi/25.
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)

        hist_label = 'Histogram of food velocity phi (rad)'
        hist_description = 'Histogram of the angular coordinate of the food velocity (rad, in the food system)'
        if redo:
            self.exp.load([name_x, name_y, 'fps'])

            self.exp.add_copy1d(
                name_to_copy=name_x, copy_name=result_velocity_phi_name, category=category,
                label='Food velocity phi (rad)',
                description='Angular coordinate of the food velocity (rad, in the food system)'
            )

            self.exp.add_copy1d(
                name_to_copy=name_x, copy_name=result_velocity_x_name, category=category,
                label='Food velocity X (rad)',
                description='X coordinate of the food velocity (rad, in the food system)'
            )
            self.exp.add_copy1d(
                name_to_copy=name_y, copy_name=result_velocity_y_name, category=category,
                label='Food velocity Y (rad)',
                description='Y coordinate of the food velocity (rad, in the food system)'
            )

            for id_exp in self.exp.characteristic_timeseries_exp_frame_index:
                fps = self.exp.get_value('fps', id_exp)

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
                self.exp.get_df(result_velocity_phi_name).loc[id_exp, :] = np.around(dvel_phi, 3)

                self.exp.get_df(result_velocity_x_name).loc[id_exp, :] = np.around(dx*fps, 3)
                self.exp.get_df(result_velocity_y_name).loc[id_exp, :] = np.around(dy*fps, 3)

            self.exp.write(result_velocity_phi_name)
            self.exp.write(result_velocity_x_name)
            self.exp.write(result_velocity_y_name)

        self.compute_hist(hist_name=hist_name, name=result_velocity_phi_name, bins=bins,
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

        else:
            self.exp.load(result_name)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter2 = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig2, ax2 = plotter2.plot(xlabel='Food direction error', ylabel='PDF', normed=True)
        plotter2.save(fig2)

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

    def compute_mm1s_food_direction_error(self, redo=False, redo_plot_indiv=False, redo_hist=False):

        name = 'food_direction_error'
        result_name = 'mm1s_' + name
        food_exit_angle_name = 'mm1s_food_exit_angle'
        vel_name = 'mm1s_food_velocity'
        time_window = 100
        category = 'FoodBase'
        self.__compute_food_direction_error(result_name, category, food_exit_angle_name, vel_name,
                                            time_window, redo, redo_hist, redo_plot_indiv)

    def compute_mm10s_food_direction_error(self, redo=False, redo_plot_indiv=False, redo_hist=False):

        name = 'food_direction_error'
        result_name = 'mm10s_' + name
        food_exit_angle_name = 'mm10s_food_exit_angle'
        vel_name = 'mm10s_food_velocity'
        time_window = 1000
        category = 'FoodBase'
        self.__compute_food_direction_error(result_name, category, food_exit_angle_name, vel_name,
                                            time_window, redo, redo_hist, redo_plot_indiv)

    def compute_mm30s_food_direction_error(self, redo=False, redo_plot_indiv=False, redo_hist=False):

        name = 'food_direction_error'
        result_name = 'mm30s_' + name
        food_exit_angle_name = 'mm30s_food_exit_angle'
        vel_name = 'mm30s_food_velocity'
        time_window = 3000
        category = 'FoodBase'
        self.__compute_food_direction_error(result_name, category, food_exit_angle_name, vel_name,
                                            time_window, redo, redo_hist, redo_plot_indiv)

    def __compute_food_direction_error(self, result_name, category, food_exit_angle_name, vel_name, time_window,
                                       redo, redo_hist, redo_plot_indiv):
        dtheta = np.pi / 12.
        bins = np.around(np.arange(-np.pi + dtheta / 2., np.pi + dtheta, dtheta), 3)
        result_label = 'Food direction error'
        result_description = 'Angle between the food-exit angle and the angular coordinate of the food trajectory, ' \
                             'both angles are smoothed by a moving mean of window '+str(time_window)+' frames'

        if redo:
            vel_name_x = vel_name+'_x'
            vel_name_y = vel_name+'_y'
            self.exp.load([food_exit_angle_name, vel_name_x, vel_name_y])

            vel = pd.DataFrame(index=self.exp.get_index(vel_name_x))
            vel['x'] = self.exp.get_df(vel_name_x)
            vel['y'] = self.exp.get_df(vel_name_y)

            vel_phi = angle_df(vel)

            tab = self.exp.get_df(food_exit_angle_name)[food_exit_angle_name] - vel_phi

            self.exp.add_copy1d(name_to_copy=vel_name_x, copy_name=result_name, category=category,
                                label=result_label, description=result_description)

            self.exp.change_values(result_name, np.around(tab, 3))

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        if redo or redo_plot_indiv:
            attachment_name = 'outside_ant_carrying_intervals'
            self.exp.load(['fps', attachment_name])

            def plot4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                fps = self.exp.get_value('fps', id_exp)
                df2 = df.loc[id_exp, :]
                df2.index = df2.index / fps

                self.exp.add_new_dataset_from_df(df=df2, name='temp', category=category, replace=True)

                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
                fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Food direction error', marker='')

                # self.exp.add_new_dataset_from_df(df=np.cos(df2), name='temp', category=category, replace=True)
                #
                # plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
                # fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Food direction error', marker='', preplot=(fig, ax), c='red')

                attachments = self.exp.get_df(attachment_name).loc[id_exp, :]
                attachments.reset_index(inplace=True)
                attachments = np.array(attachments)

                colors = plotter.color_object.create_cmap('hot_r', set(list(attachments[:, 0])))
                for id_ant, frame, inter in attachments:
                    ax.axvline(frame / fps, c=colors[str(id_ant)], alpha=0.5)

                ax.grid()
                ax.axhline(0, c='k', label='y=0')
                ax.axhline(np.pi / 2., c='k', ls=':', label='|y|=0.5')
                ax.axhline(-np.pi / 2., c='k', ls=':')
                ax.legend(loc=0)
                ax.set_ylim((-np.pi, np.pi))
                plotter.save(fig, name=id_exp, sub_folder=result_name)

            self.exp.groupby(result_name, id_exp_name, plot4each_group)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter2 = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig2, ax2 = plotter2.plot(xlabel='Food direction error', ylabel='PDF', normed=True)
        plotter2.save(fig2)

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

    def compute_food_speed(self, redo=False, redo_hist=False):
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
        fig, ax = plotter.plot(yscale='log', xlabel=r'$\dot v (mm/s)$', ylabel='PDF',
                               normed=True, label_suffix='s')
        plotter.save(fig)

    def compute_food_speed_evol(self, redo=False):
        name = 'food_speed'
        result_name = name+'_hist_evol'

        bins = np.arange(0, 200, 1)
        frame_intervals = np.arange(0, 5., 0.5)*60*100

        if redo:
            self.exp.load(name)
            self.exp.operation(name, lambda x: np.abs(x))
            self.exp.hist1d_time_evolution(name_to_hist=name, frame_intervals=frame_intervals, bins=bins,
                                           result_name=result_name, category='FoodBase',
                                           label='Food speed distribution over time (rad)',
                                           description='Histogram of the instantaneous speed '
                                                       'of the food trajectory over time (rad)')
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(yscale='log',  xlabel=r'$\dot v (mm/s)$', ylabel='PDF',
                               normed=True, label_suffix='s', marker='', lw=1)
        ax.set_xlim((0, 80))
        plotter.save(fig)

    def compute_mm1s_food_velocity_phi(self):
        name = 'food_velocity_phi'
        category = 'FoodBase'
        time_window = 100

        self.exp.load(name)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name, time_window=time_window,
                                                  result_name='mm1s_' + name, category=category)

        self.exp.write('mm1s_' + name)

    def compute_mm1s_food_velocity_vector(self):
        name = 'food_velocity'
        name_x = name+'_x'
        name_y = name+'_y'
        category = 'FoodBase'
        time_window = 100

        self.exp.load([name_x, name_y])
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_x, time_window=time_window,
                                                  result_name='mm1s_' + name_x, category=category)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_y, time_window=time_window,
                                                  result_name='mm1s_' + name_y, category=category)

        self.exp.write('mm1s_' + name_x)
        self.exp.write('mm1s_' + name_y)

    def compute_mm10s_food_velocity_vector(self):
        name = 'food_velocity'
        name_x = name+'_x'
        name_y = name+'_y'
        category = 'FoodBase'
        time_window = 1000

        self.exp.load([name_x, name_y])
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_x, time_window=time_window,
                                                  result_name='mm10s_' + name_x, category=category)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_y, time_window=time_window,
                                                  result_name='mm10s_' + name_y, category=category)

        self.exp.write('mm10s_' + name_x)
        self.exp.write('mm10s_' + name_y)

    def compute_mm30s_food_velocity_vector(self):
        name = 'food_velocity'
        name_x = name+'_x'
        name_y = name+'_y'
        category = 'FoodBase'
        time_window = 3000

        self.exp.load([name_x, name_y])
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_x, time_window=time_window,
                                                  result_name='mm30s_' + name_x, category=category)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_y, time_window=time_window,
                                                  result_name='mm30s_' + name_y, category=category)

        self.exp.write('mm30s_' + name_x)
        self.exp.write('mm30s_' + name_y)

    def compute_mm60s_food_velocity_vector(self):
        name = 'food_velocity'
        name_x = name+'_x'
        name_y = name+'_y'
        category = 'FoodBase'
        time_window = 6000

        self.exp.load([name_x, name_y])
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_x, time_window=time_window,
                                                  result_name='mm60s_' + name_x, category=category)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_y, time_window=time_window,
                                                  result_name='mm60s_' + name_y, category=category)

        self.exp.write('mm60s_' + name_x)
        self.exp.write('mm60s_' + name_y)

    def compute_mm1s_food_velocity_vector_length(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        time = '1'
        result_name = 'mm' + time + 's_food_velocity_vector_length'
        category = 'FoodBase'
        bins = np.arange(0, 20, 0.5)

        self.__compute_indiv_food_velocity_vector_length(category, redo, result_name, time)

        self.__plot_indiv_food_velocity_vector_length(result_name, category, redo, redo_plot_indiv)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='length (mm)', ylabel='PDF', normed=True)
        plotter.save(fig)

    def compute_mm10s_food_velocity_vector_length(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        time = '10'
        result_name = 'mm' + time + 's_food_velocity_vector_length'
        category = 'FoodBase'
        bins = np.arange(0, 15, 0.5)

        self.__compute_indiv_food_velocity_vector_length(category, redo, result_name, time)

        self.__plot_indiv_food_velocity_vector_length(result_name, category, redo, redo_plot_indiv)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='length (mm)', ylabel='PDF', normed=True)
        plotter.save(fig)

    def compute_mm30s_food_velocity_vector_length(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        time = '30'
        result_name = 'mm' + time + 's_food_velocity_vector_length'
        category = 'FoodBase'
        bins = np.arange(0, 15, 0.5)

        self.__compute_indiv_food_velocity_vector_length(category, redo, result_name, time)

        self.__plot_indiv_food_velocity_vector_length(result_name, category, redo, redo_plot_indiv)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='length (mm)', ylabel='PDF', normed=True)
        plotter.save(fig)

        df_m = self.exp.groupby(result_name, id_exp_name, lambda x: np.max(x))
        self.exp.get_data_object(result_name).df = self.exp.get_df(result_name) / df_m
        self.compute_hist(name=result_name, bins=np.arange(0, 1.1, 0.05), hist_name='temp',
                          redo=True, redo_hist=True, write=False)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
        fig, ax = plotter.plot(xlabel='length (mm)', ylabel='PDF', normed=True)
        plotter.save(fig, name=hist_name, suffix='norm')

    def compute_mm60s_food_velocity_vector_length(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        time = '60'
        result_name = 'mm' + time + 's_food_velocity_vector_length'
        category = 'FoodBase'
        bins = np.arange(0, 15, 0.5)

        self.__compute_indiv_food_velocity_vector_length(category, redo, result_name, time)

        self.__plot_indiv_food_velocity_vector_length(result_name, category, redo, redo_plot_indiv)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='length (mm)', ylabel='PDF', normed=True)
        plotter.save(fig)

    def __plot_indiv_food_velocity_vector_length(self, result_name, category, redo, redo_plot_indiv):
        if redo or redo_plot_indiv:
            attachment_name = 'outside_ant_carrying_intervals'
            self.exp.load(['fps', attachment_name])

            def plot4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                fps = self.exp.get_value('fps', id_exp)
                df3 = df.loc[id_exp, :]
                df3.index = df3.index / fps

                self.exp.add_new_dataset_from_df(df=df3, name='temp', category=category, replace=True)
                # m = self.exp.temp.df.max()
                # self.exp.get_data_object('temp').df = self.exp.get_df('temp') / m

                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
                fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Vector length', marke='')
                # ax.set_ylim((0, 1))

                attachments = self.exp.get_df(attachment_name).loc[id_exp, :]
                attachments.reset_index(inplace=True)
                attachments = np.array(attachments)

                colors = plotter.color_object.create_cmap('hot_r', set(list(attachments[:, 0])))
                for id_ant, frame, inter in attachments:
                    ax.axvline(frame / fps, c=colors[str(id_ant)], alpha=0.5)
                ax.grid()
                plotter.save(fig, name=id_exp, sub_folder=result_name)

            self.exp.groupby(result_name, id_exp_name, plot4each_group)

    def __compute_indiv_food_velocity_vector_length(self, category, redo, result_name, time):
        if redo:
            name_x = 'mm' + time + 's_food_velocity_x'
            name_y = 'mm' + time + 's_food_velocity_y'

            self.exp.load([name_x, name_y])

            self.exp.add_copy1d(name_to_copy=name_x, copy_name=result_name, category=category,
                                label='Food velocity vector length (mm)',
                                description='Length of the sum over ' + time + ' seconds of the velocity vector')

            def compute_length4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                print(id_exp)
                x = self.exp.get_df(name_x).loc[pd.IndexSlice[id_exp, :], :]
                y = self.exp.get_df(name_y).loc[pd.IndexSlice[id_exp, :], :]

                lengths = np.sqrt(x.loc[:, name_x] ** 2 + y.loc[:, name_y] ** 2)
                self.exp.get_df(result_name).loc[id_exp, :] = np.around(lengths, 3)

            self.exp.groupby(result_name, id_exp_name, compute_length4each_group)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

    def __compute_indiv_dotproduct_food_velocity_exit(self, category, redo, result_name, time):
        if redo:
            if time is None:

                vel_name_x = 'food_velocity_x'
                vel_name_y = 'food_velocity_y'
                angle_food_exit_name = 'food_exit_angle'
                self.exp.load([vel_name_x, vel_name_y, angle_food_exit_name])

                self.exp.add_copy1d(name_to_copy=vel_name_x, copy_name=result_name, category=category,
                                    label='Dot product between food velocity and food-exit vector',
                                    description='Dot product between the food-exit vector and food velocity vector')

            else:

                vel_name_x = 'mm' + time + 's_food_velocity_x'
                vel_name_y = 'mm' + time + 's_food_velocity_y'
                angle_food_exit_name = 'mm' + time + 's_food_exit_angle'
                self.exp.load([vel_name_x, vel_name_y, angle_food_exit_name])

                self.exp.add_copy1d(name_to_copy=vel_name_x, copy_name=result_name, category=category,
                                    label='Dot product between food velocity and food-exit vector',
                                    description='Dot product between the food-exit vector and food velocity vector '
                                                'smoothed by a moving mean of window' + time + ' seconds')

            def compute_length4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                print(id_exp)

                vel_x = self.exp.get_df(vel_name_x).loc[pd.IndexSlice[id_exp, :], :]
                vel_y = self.exp.get_df(vel_name_y).loc[pd.IndexSlice[id_exp, :], :]
                vel = pd.DataFrame(index=vel_x.index)
                vel['x'] = vel_x
                vel['y'] = vel_y

                food_exit_phi = self.exp.get_df(angle_food_exit_name).loc[pd.IndexSlice[id_exp, :], :]
                food_exit = pd.DataFrame(index=vel_x.index)
                food_exit['x'] = np.cos(food_exit_phi)
                food_exit['y'] = np.sin(food_exit_phi)

                dot_prod = dot2d_df(vel, food_exit)/distance_df(vel)
                self.exp.get_df(result_name).loc[id_exp, :] = np.around(dot_prod, 3)

            self.exp.groupby(result_name, id_exp_name, compute_length4each_group)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

    def __plot_indiv_dotproduct_food_velocity_exit(self, result_name, category, redo, redo_plot_indiv):
        if redo or redo_plot_indiv:
            attachment_name = 'outside_ant_carrying_intervals'
            self.exp.load(['fps', attachment_name])

            def plot4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                fps = self.exp.get_value('fps', id_exp)
                df3 = df.loc[id_exp, :]
                df3.index = df3.index / fps

                self.exp.add_new_dataset_from_df(df=df3, name='temp', category=category, replace=True)

                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
                fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Dot product', marker='')

                attachments = self.exp.get_df(attachment_name).loc[id_exp, :]
                attachments.reset_index(inplace=True)
                attachments = np.array(attachments)

                colors = plotter.color_object.create_cmap('hot_r', set(list(attachments[:, 0])))
                for id_ant, frame, inter in attachments:
                    ax.axvline(frame / fps, c=colors[str(id_ant)], alpha=0.5)

                ax.grid()
                ax.axhline(0, c='k', label='y=0')
                ax.axhline(0.5, c='k', ls=':', label='|y|=0.5')
                ax.axhline(-0.5, c='k', ls=':')
                ax.legend(loc=0)
                ax.set_ylim((-1.1, 1.1))
                plotter.save(fig, name=id_exp, sub_folder=result_name)

            self.exp.groupby(result_name, id_exp_name, plot4each_group)

    def compute_dotproduct_food_velocity_exit(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        result_name = 'dotproduct_food_velocity_exit'
        category = 'FoodBase'

        self.__compute_indiv_dotproduct_food_velocity_exit(category, redo, result_name)
        self.__plot_indiv_dotproduct_food_velocity_exit(result_name, category, redo, redo_plot_indiv)
        self.__plot_hist_dotproduct_vel_exit(result_name, redo, redo_hist)

    def compute_mm1s_dotproduct_food_velocity_exit(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        time = '1'
        result_name = 'mm' + time + 's_dotproduct_food_velocity_exit'
        category = 'FoodBase'

        self.__compute_indiv_dotproduct_food_velocity_exit(category, redo, result_name, time)
        self.__plot_indiv_dotproduct_food_velocity_exit(result_name, category, redo, redo_plot_indiv)
        self.__plot_hist_dotproduct_vel_exit(result_name, redo, redo_hist)

    def __plot_hist_dotproduct_vel_exit(self, result_name,  redo, redo_hist):
        bins = np.arange(-1, 1.1, 0.1)
        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='Dot product', ylabel='PDF', normed=True, xlim=(-1.1, 1.1))
        plotter.save(fig)

    def compute_mm10s_dotproduct_food_velocity_exit(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        time = '10'
        result_name = 'mm' + time + 's_dotproduct_food_velocity_exit'
        category = 'FoodBase'

        self.__compute_indiv_dotproduct_food_velocity_exit(category, redo, result_name, time)
        self.__plot_indiv_dotproduct_food_velocity_exit(result_name, category, redo, redo_plot_indiv)
        self.__plot_hist_dotproduct_vel_exit(result_name, redo, redo_hist)

    def compute_mm30s_dotproduct_food_velocity_exit(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        time = '30'
        result_name = 'mm' + time + 's_dotproduct_food_velocity_exit'
        category = 'FoodBase'

        self.__compute_indiv_dotproduct_food_velocity_exit(category, redo, result_name, time)
        self.__plot_indiv_dotproduct_food_velocity_exit(result_name, category, redo, redo_plot_indiv)
        self.__plot_hist_dotproduct_vel_exit(result_name, redo, redo_hist)

    def compute_mm60s_dotproduct_food_velocity_exit(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        time = '60'
        result_name = 'mm' + time + 's_dotproduct_food_velocity_exit'
        category = 'FoodBase'

        self.__compute_indiv_dotproduct_food_velocity_exit(category, redo, result_name, time)
        self.__plot_indiv_dotproduct_food_velocity_exit(result_name, category, redo, redo_plot_indiv)
        self.__plot_hist_dotproduct_vel_exit(result_name, redo, redo_hist)

    def compute_w30s_entropy_mm1s_food_velocity_phi_indiv_evol(self, redo=False, redo_indiv_plot=False):
        mm = 1
        w = 30

        result_name = 'w'+str(w)+'s_entropy_mm1s_food_velocity_phi_indiv_evol'
        category = 'FoodBase'

        time_intervals = np.arange(0, 10*60)
        dtheta = np.pi / 12.
        bins = np.around(np.arange(-np.pi + dtheta/2., np.pi + dtheta, dtheta), 3)

        self.__compute_food_velocity_entropy(w, bins, category, mm, redo, redo_indiv_plot, result_name, time_intervals)

    def compute_w10s_entropy_mm1s_food_velocity_phi_indiv_evol(self, redo=False, redo_indiv_plot=False):
        mm = 1
        w = 10

        result_name = 'w'+str(w)+'s_entropy_mm1s_food_velocity_phi_indiv_evol'
        category = 'FoodBase'

        time_intervals = np.arange(0, 10*60, 1)
        dtheta = np.pi / 12.
        bins = np.around(np.arange(-np.pi + dtheta/2., np.pi + dtheta, dtheta), 3)

        self.__compute_food_velocity_entropy(w, bins, category, mm, redo, redo_indiv_plot, result_name, time_intervals)

    def compute_w1s_entropy_mm1s_food_velocity_phi_indiv_evol(self, redo=False, redo_indiv_plot=False):
        mm = 1
        w = 1

        result_name = 'w'+str(w)+'s_entropy_mm1s_food_velocity_phi_indiv_evol'
        category = 'FoodBase'

        time_intervals = np.arange(0, 10*60, 1)
        dtheta = np.pi / 12.
        bins = np.around(np.arange(-np.pi + dtheta/2., np.pi + dtheta, dtheta), 3)

        self.__compute_food_velocity_entropy(w, bins, category, mm, redo, redo_indiv_plot, result_name, time_intervals)

    def __compute_food_velocity_entropy(self,
                                        w, bins, category, mm, redo, redo_indiv_plot, result_name, time_intervals):
        if redo:
            vel_phi_name = 'mm' + str(mm) + 's_food_velocity_phi'
            self.exp.load([vel_phi_name, 'fps'])
            self.exp.add_new_empty_dataset(name=result_name, index_names='time',
                                           column_names=self.exp.id_exp_list, index_values=time_intervals,
                                           category=category, label='Evolution of the entropy of the food velocity phi',
                                           description='Time evolution of the entropy of the distribution'
                                                       ' of the angular coordinate'
                                                       ' of food velocity for each experiment')

            def compute_entropy4each_group(df: pd.DataFrame):
                exp = df.index.get_level_values(id_exp_name)[0]
                print(exp)
                fps0 = self.exp.get_value('fps', exp)
                frame0 = df.index.get_level_values(id_frame_name).min()

                w0 = w * fps0
                for time in time_intervals:
                    f0 = time * fps0 - w0 + frame0
                    f1 = time * fps0 + w0 + frame0

                    vel = df.loc[pd.IndexSlice[exp, f0:f1], :]
                    hist = np.histogram(vel, bins, normed=False)
                    hist = hist[0] / np.sum(hist[0])
                    if len(vel) != 0:
                        entropy = np.around(get_entropy(hist), 3)
                        self.exp.change_value(result_name, (time, exp), entropy)

            self.exp.groupby(vel_phi_name, id_exp_name, compute_entropy4each_group)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        if redo or redo_indiv_plot:

            attachment_name = 'outside_ant_carrying_intervals'
            self.exp.load(['fps', attachment_name])

            for id_exp in self.exp.get_df(result_name).columns:
                id_exp = int(id_exp)
                fps = self.exp.get_value('fps', id_exp)

                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name=id_exp)
                fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Entropy',
                                       title_prefix='Exp ' + str(id_exp) + ': ')

                attachments = self.exp.get_df(attachment_name).loc[id_exp, :]
                attachments.reset_index(inplace=True)
                attachments = np.array(attachments)

                colors = plotter.color_object.create_cmap('hot_r', set(list(attachments[:, 0])))
                for id_ant, frame, inter in attachments:
                    ax.axvline(frame / fps, c=colors[str(id_ant)], alpha=0.5)
                ax.grid()
                ax.set_ylim((0.5, 4))
                plotter.save(fig, name=id_exp, sub_folder=result_name)

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

    def __reindexing_food_xy(self, id_ants, idxs):
        df_d = self.exp.food_xy.df.copy()
        df_d = df_d.reindex(idxs)
        df_d[id_ant_name] = id_ants
        df_d.reset_index(inplace=True)
        df_d.columns = [id_exp_name, id_frame_name, 'x', 'y', id_ant_name]
        df_d.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)
        return df_d

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
