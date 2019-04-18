import numpy as np
import pandas as pd

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name
from Tools.MiscellaneousTools.Geometry import angle, dot2d_df, distance_df
from Tools.Plotter.Plotter import Plotter


class AnalyseFoodVelocity(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'FoodVelocity'

    def compute_food_velocity(self, redo=False, redo_hist=False):
        name_x = 'mm10_food_x'
        name_y = 'mm10_food_y'

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
                name_to_copy=name_x, copy_name=result_velocity_phi_name, category=self.category,
                label='Food velocity phi (rad)',
                description='Angular coordinate of the food velocity (rad, in the food system)'
            )

            self.exp.add_copy1d(
                name_to_copy=name_x, copy_name=result_velocity_x_name, category=self.category,
                label='Food velocity X (rad)',
                description='X coordinate of the food velocity (rad, in the food system)'
            )
            self.exp.add_copy1d(
                name_to_copy=name_y, copy_name=result_velocity_y_name, category=self.category,
                label='Food velocity Y (rad)',
                description='Y coordinate of the food velocity (rad, in the food system)'
            )

            for id_exp in self.exp.characteristic_timeseries_exp_frame_index:
                fps = self.exp.get_value('fps', id_exp)

                dx = np.array(self.exp.get_df(name_x).loc[id_exp, :]).ravel()
                dx1 = dx[1].copy()
                dx2 = dx[-2].copy()
                dx[1:-1] = (dx[2:] - dx[:-2]) / 2.
                dx[0] = dx1 - dx[0]
                dx[-1] = dx[-1] - dx2

                dy = np.array(self.exp.get_df(name_y).loc[id_exp, :]).ravel()
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
                                           result_name=result_name, category=self.category,
                                           label='Food velocity phi distribution over time (rad)',
                                           description='Histogram of the absolute value of the angular coordinate'
                                                       ' of the velocity of the food trajectory over time (rad)')
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\varphi$', ylabel='PDF', normed=True, label_suffix='s')
        plotter.save(fig)

    def compute_dotproduct_food_velocity_exit(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        result_name = 'dotproduct_food_velocity_exit'

        self.__compute_indiv_dotproduct_food_velocity_exit(self.category, redo, result_name)
        self.__plot_indiv_dotproduct_food_velocity_exit(result_name, self.category, redo, redo_plot_indiv)
        self.__plot_hist_dotproduct_vel_exit(result_name, redo, redo_hist)

    def compute_mm1s_dotproduct_food_velocity_exit(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        time = '1'
        result_name = 'mm' + time + 's_dotproduct_food_velocity_exit'

        self.__compute_indiv_dotproduct_food_velocity_exit(self.category, redo, result_name, time)
        self.__plot_indiv_dotproduct_food_velocity_exit(result_name, self.category, redo, redo_plot_indiv)
        self.__plot_hist_dotproduct_vel_exit(result_name, redo, redo_hist)

    def compute_mm10s_dotproduct_food_velocity_exit(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        time = '10'
        result_name = 'mm' + time + 's_dotproduct_food_velocity_exit'

        self.__compute_indiv_dotproduct_food_velocity_exit(self.category, redo, result_name, time)
        self.__plot_indiv_dotproduct_food_velocity_exit(result_name, self.category, redo, redo_plot_indiv)
        self.__plot_hist_dotproduct_vel_exit(result_name, redo, redo_hist)

    def compute_mm30s_dotproduct_food_velocity_exit(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        time = '30'
        result_name = 'mm' + time + 's_dotproduct_food_velocity_exit'

        self.__compute_indiv_dotproduct_food_velocity_exit(self.category, redo, result_name, time)
        self.__plot_indiv_dotproduct_food_velocity_exit(result_name, self.category, redo, redo_plot_indiv)
        self.__plot_hist_dotproduct_vel_exit(result_name, redo, redo_hist)

    def __compute_indiv_dotproduct_food_velocity_exit(self, category, redo, result_name, time=None):
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

                dot_prod = dot2d_df(vel, food_exit) / distance_df(vel)
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

    def __plot_hist_dotproduct_vel_exit(self, result_name,  redo, redo_hist):
        bins = np.arange(-1, 1.1, 0.1)
        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='Dot product', ylabel='PDF', normed=True, xlim=(-1.1, 1.1))
        plotter.save(fig)

    def compute_mm1s_food_velocity_phi(self):
        name = 'food_velocity_phi'
        time_window = 100

        self.exp.load(name)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name, time_window=time_window,
                                                  result_name='mm1s_' + name, category=self.category)

        self.exp.write('mm1s_' + name)

    def compute_mm1s_food_velocity_vector(self):
        name = 'food_velocity'
        name_x = name+'_x'
        name_y = name+'_y'
        time_window = 100

        self.exp.load([name_x, name_y])
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_x, time_window=time_window,
                                                  result_name='mm1s_' + name_x, category=self.category)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_y, time_window=time_window,
                                                  result_name='mm1s_' + name_y, category=self.category)

        self.exp.write('mm1s_' + name_x)
        self.exp.write('mm1s_' + name_y)

    def compute_mm10s_food_velocity_vector(self):
        name = 'food_velocity'
        name_x = name+'_x'
        name_y = name+'_y'
        time_window = 1000

        self.exp.load([name_x, name_y])
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_x, time_window=time_window,
                                                  result_name='mm10s_' + name_x, category=self.category)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_y, time_window=time_window,
                                                  result_name='mm10s_' + name_y, category=self.category)

        self.exp.write('mm10s_' + name_x)
        self.exp.write('mm10s_' + name_y)

    def compute_mm30s_food_velocity_vector(self):
        name = 'food_velocity'
        name_x = name+'_x'
        name_y = name+'_y'
        time_window = 3000

        self.exp.load([name_x, name_y])
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_x, time_window=time_window,
                                                  result_name='mm30s_' + name_x, category=self.category)
        self.exp.moving_mean4exp_frame_indexed_1d(name_to_average=name_y, time_window=time_window,
                                                  result_name='mm30s_' + name_y, category=self.category)

        self.exp.write('mm30s_' + name_x)
        self.exp.write('mm30s_' + name_y)

    def compute_mm1s_food_velocity_vector_length(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        time = '1'
        result_name = 'mm' + time + 's_food_velocity_vector_length'
        bins = np.arange(0, 20, 0.5)

        self.__compute_indiv_food_velocity_vector_length(self.category, redo, result_name, time)

        self.__plot_indiv_food_velocity_vector_length(result_name, self.category, redo, redo_plot_indiv)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='length (mm)', ylabel='PDF', normed=True)
        plotter.save(fig)

    def compute_mm10s_food_velocity_vector_length(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        time = '10'
        result_name = 'mm' + time + 's_food_velocity_vector_length'
        bins = np.arange(0, 15, 0.5)

        self.__compute_indiv_food_velocity_vector_length(self.category, redo, result_name, time)

        self.__plot_indiv_food_velocity_vector_length(result_name, self.category, redo, redo_plot_indiv)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='length (mm)', ylabel='PDF', normed=True)
        plotter.save(fig)

    def compute_mm30s_food_velocity_vector_length(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        time = '30'
        result_name = 'mm' + time + 's_food_velocity_vector_length'
        bins = np.arange(0, 15, 0.5)

        self.__compute_indiv_food_velocity_vector_length(self.category, redo, result_name, time)

        self.__plot_indiv_food_velocity_vector_length(result_name, self.category, redo, redo_plot_indiv)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='length (mm)', ylabel='PDF', normed=True)
        plotter.save(fig)

        df_m = self.exp.groupby(result_name, id_exp_name, lambda x: np.max(x))
        self.exp.get_data_object(result_name).df = self.exp.get_df(result_name) / df_m
        self.compute_hist(name=result_name, bins=np.arange(0, 1.1, 0.05), hist_name='temp0',
                          redo=True, redo_hist=True, write=False)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp0'))
        fig, ax = plotter.plot(xlabel='length (mm)', ylabel='PDF', normed=True)
        plotter.save(fig, name=hist_name, suffix='norm')

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
