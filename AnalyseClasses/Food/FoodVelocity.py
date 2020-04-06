import numpy as np
import pandas as pd
import pylab as pb

import Tools.MiscellaneousTools.Geometry as Geo

from Tools.MiscellaneousTools import Fits
from scipy import interpolate
from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name
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

            def get_velocity4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                fps = self.exp.get_value('fps', id_exp)

                dx = np.array(df.loc[id_exp, :]).ravel()
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

                dvel_phi = Geo.angle(np.array(list(zip(dx, dy))))
                self.exp.get_df(result_velocity_phi_name).loc[id_exp, :] = np.around(dvel_phi, 6)

                self.exp.get_df(result_velocity_x_name).loc[id_exp, :] = np.around(dx*fps, 3)
                self.exp.get_df(result_velocity_y_name).loc[id_exp, :] = np.around(dy*fps, 3)

            self.exp.groupby(name_x, id_exp_name, get_velocity4each_group)

            self.exp.write(result_velocity_phi_name)
            self.exp.write(result_velocity_x_name)
            self.exp.write(result_velocity_y_name)

        self.compute_hist(hist_name=hist_name, name=result_velocity_phi_name, bins=bins,
                          hist_label=hist_label, hist_description=hist_description, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel=r'$\varphi$', ylabel='PDF', normed=True)
        plotter.save(fig)

    def compute_food_velocity_phi_hist_evol(self, redo=False):
        name = 'food_velocity_phi'
        result_name = name+'_hist_evol'
        init_frame_name = 'food_first_frame'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        dx = 0.25
        start_frame_intervals = np.arange(0, 3.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        if redo:
            self.exp.load(name)

            self.change_first_frame(name, init_frame_name)

            self.exp.operation(name, lambda a: np.abs(a))
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                      end_frame_intervals=end_frame_intervals, bins=bins,
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

    def compute_food_velocity_phi_hist_evol_around_first_outside_attachment(self, redo=False):
        name = 'food_velocity_phi'
        result_name = name+'_hist_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        dx = 0.25
        start_frame_intervals = np.arange(-1, 3.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        if redo:
            self.exp.load(name)

            self.change_first_frame(name, init_frame_name)

            self.exp.operation(name, lambda a: np.abs(a))
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                      end_frame_intervals=end_frame_intervals, bins=bins,
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

    def compute_food_velocity_phi_variance_evol(self, redo=False):
        name = 'food_velocity_phi'
        result_name = name + '_var_evol'
        init_frame_name = 'food_first_frame'

        dx = 0.1
        dx2 = 0.01
        start_frame_intervals = np.arange(0, 3.5, dx2)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        label = 'Variance of the food velocity phi over time'
        description = 'Variance of the angular coordinate of the food velocity'

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
            xlabel='Time (s)', ylabel=r'Variance $\sigma^2$',
            label_suffix='s', label=r'$\sigma^2$', title='', marker='')
        plotter.plot_fit(typ='exp', preplot=(fig, ax), window=[0, 400], cst=(-0.01, .1, .1))
        plotter.save(fig)

    def compute_food_velocity_phi_variance_evol_around_first_outside_attachment(self, redo=False):
        name = 'food_velocity_phi'
        result_name = name + '_var_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.1
        dx2 = 0.01
        start_frame_intervals = np.arange(-1, 3.5, dx2)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        label = 'Variance of the food velocity phi over time'
        description = 'Variance of the angular coordinate of the food velocity'

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
            xlabel='Time (s)', ylabel=r'Variance $\sigma^2$',
            label_suffix='s', label=r'$\sigma^2$', title='', marker='')
        plotter.plot_fit(typ='exp', preplot=(fig, ax), window=[90, 400], cst=(-0.01, .1, .1))
        plotter.draw_vertical_line(ax)
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

                dot_prod = Geo.dot2d_df(vel, food_exit) / Geo.distance_df(vel)
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

    def compute_mm10_food_velocity_phi(self):
        name = 'food_velocity_phi'
        time_window = 10

        self.exp.load(name)
        self.exp.rolling_mean(name_to_average=name, window=time_window,
                              result_name='mm10_' + name, category=self.category, is_angle=True)

        self.exp.write('mm10_' + name)

    def compute_mm1s_food_velocity_phi(self):
        name = 'food_velocity_phi'
        time_window = 100

        self.exp.load(name)
        self.exp.rolling_mean(name_to_average=name, window=time_window,
                              result_name='mm1s_' + name, category=self.category, is_angle=True)

        self.exp.write('mm1s_' + name)

    def compute_mm10_food_velocity_vector(self):
        name = 'food_velocity'
        name_x = name+'_x'
        name_y = name+'_y'
        time_window = 10

        self.exp.load([name_x, name_y])
        self.exp.rolling_mean(name_to_average=name_x, window=time_window,
                              result_name='mm10_' + name_x, category=self.category, is_angle=False)
        self.exp.rolling_mean(name_to_average=name_y, window=time_window,
                              result_name='mm10_' + name_y, category=self.category, is_angle=False)

        self.exp.write('mm10_' + name_x)
        self.exp.write('mm10_' + name_y)

    def compute_mm20_food_velocity_vector(self):
        name = 'food_velocity'
        name_x = name+'_x'
        name_y = name+'_y'
        time_window = 20

        self.exp.load([name_x, name_y])
        self.exp.rolling_mean(name_to_average=name_x, window=time_window,
                              result_name='mm20_' + name_x, category=self.category, is_angle=False)
        self.exp.rolling_mean(name_to_average=name_y, window=time_window,
                              result_name='mm20_' + name_y, category=self.category, is_angle=False)

        self.exp.write('mm20_' + name_x)
        self.exp.write('mm20_' + name_y)

    def compute_mm1s_food_velocity_vector(self):
        name = 'food_velocity'
        name_x = name+'_x'
        name_y = name+'_y'
        time_window = 100

        self.exp.load([name_x, name_y])
        self.exp.rolling_mean(name_to_average=name_x, window=time_window,
                              result_name='mm1s_' + name_x, category=self.category, is_angle=False)
        self.exp.rolling_mean(name_to_average=name_y, window=time_window,
                              result_name='mm1s_' + name_y, category=self.category, is_angle=False)

        self.exp.write('mm1s_' + name_x)
        self.exp.write('mm1s_' + name_y)

    def compute_mm10s_food_velocity_vector(self):
        name = 'food_velocity'
        name_x = name+'_x'
        name_y = name+'_y'
        time_window = 1000

        self.exp.load([name_x, name_y])
        self.exp.rolling_mean(name_to_average=name_x, window=time_window,
                              result_name='mm10s_' + name_x, category=self.category, is_angle=False)
        self.exp.rolling_mean(name_to_average=name_y, window=time_window,
                              result_name='mm10s_' + name_y, category=self.category, is_angle=False)

        self.exp.write('mm10s_' + name_x)
        self.exp.write('mm10s_' + name_y)

    def compute_mm30s_food_velocity_vector(self):
        name = 'food_velocity'
        name_x = name+'_x'
        name_y = name+'_y'
        time_window = 3000

        self.exp.load([name_x, name_y])
        self.exp.rolling_mean(name_to_average=name_x, window=time_window,
                              result_name='mm30s_' + name_x, category=self.category, is_angle=False)
        self.exp.rolling_mean(name_to_average=name_y, window=time_window,
                              result_name='mm30s_' + name_y, category=self.category, is_angle=False)

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

    def compute_food_orientation_speed(self, redo=False, redo_hist=False):

        velocity_phi_name = 'food_velocity_phi'
        result_name = 'food_orientation_speed'

        hist_name = result_name+'_hist'
        dtheta = np.pi/25.
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)

        label = 'Food orientation speed (rad/s)'
        hist_label = 'Histogram of the %s' % label
        description = 'Speed of the angular coordinate of the food velocity' \
                      ' (rad, in the food system)'
        hist_description = 'Histogram of the %s' % description
        if redo:
            self.exp.load([velocity_phi_name, 'fps'])

            self.exp.add_copy1d(
                name_to_copy=velocity_phi_name, copy_name=result_name, category=self.category,
                label=label, description=description)

            def get_velocity4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                fps = self.exp.get_value('fps', id_exp)

                dx = np.array(df.loc[id_exp, :]).ravel()
                dx1 = dx[1].copy()
                dx2 = dx[-2].copy()
                dx[1:-1] = Geo.angle_distance(dx[:-2], dx[2:]) / 2.
                dx[0] = Geo.angle_distance(dx[0], dx1)
                dx[-1] = Geo.angle_distance(dx2, dx[-1])

                self.exp.get_df(result_name).loc[id_exp, :] = np.c_[np.around(dx*fps, 6)]

            self.exp.groupby(velocity_phi_name, id_exp_name, get_velocity4each_group)

            self.exp.write(result_name)

        self.compute_hist(hist_name=hist_name, name=result_name, bins=bins,
                          hist_label=hist_label, hist_description=hist_description, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel=r'$\dot\varphi$', ylabel='PDF', normed=True)
        plotter.save(fig)

    def compute_food_velocity_phi_diff(self, redo, redo_hist=False):
        result_name = 'mm1s_food_velocity_phi_diff'

        if redo:
            velocity_phi_name = 'mm1s_food_velocity_phi'
            food_speed_name = 'mm10_food_speed'
            self.exp.load([velocity_phi_name, food_speed_name])

            label = 'Food orientation time difference'
            description = 'Food orientation difference'

            self.exp.add_copy(
                velocity_phi_name, result_name, category=self.category, label=label, description=description)

            def get_diff4each(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                print(id_exp)
                v = self.exp.get_df(food_speed_name).loc[id_exp, :]
                v = np.array(v.reindex(df.loc[id_exp, :].index)).ravel()

                orient = df.values.ravel()
                mask = np.where(~np.isnan(orient))[0]
                d_orient = Geo.angle_distance(orient[mask[1:]], orient[mask[:-1]])
                orient[:] = np.nan
                orient[mask[1:]] = d_orient

                orient[v < 2] = np.nan

                self.exp.get_df(result_name).loc[id_exp, :] = np.c_[np.around(orient, 6)]

            self.exp.groupby(result_name, id_exp_name, get_diff4each)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        dtheta = 0.01
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)
        hist_name = self.compute_hist(
            name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel=r'$\varphi$', ylabel='PDF', normed=True)
        plotter.save(fig)

        init_frame_name = 'first_leading_attachment_time_of_outside_ant'
        self.change_first_frame(result_name, init_frame_name)
        print(float(np.var(self.exp.get_df(result_name).loc[pd.IndexSlice[:, :0], :])))

    def compute_var_orientation(self):

        self.exp.load(['mm10_food_x', 'mm10_food_y'])
        init_frame_name = 'first_attachment_time_of_outside_ant'
        self.change_first_frame('mm10_food_x', init_frame_name)
        self.change_first_frame('mm10_food_y', init_frame_name)
        dx = 3
        list_dorient = []
        list_orient = []
        for id_exp in range(1, 61):
            tab_x = self.exp.get_df('mm10_food_x').loc[pd.IndexSlice[id_exp, -1500:0], :].values.ravel()
            tab_y = self.exp.get_df('mm10_food_y').loc[pd.IndexSlice[id_exp, -1500:0], :].values.ravel()

            mask = np.where(np.abs(np.diff(tab_x)) + np.abs(np.diff(tab_y)) > 0)
            tab_x = tab_x[mask]
            tab_y = tab_y[mask]

            # traj_length = np.sum(np.sqrt(np.diff(tab_x)**2+np.diff(tab_y)**2))
            spline = interpolate.splprep([tab_x, tab_y], s=1)
            x_i, y_i = interpolate.splev(spline[1], spline[0])
            # x_i, y_i = interpolate.splev(np.linspace(0, 1, traj_length*10), spline[0])
            # traj_length = np.sum(np.sqrt(np.diff(x_i)**2+np.diff(y_i)**2))
            #
            # di = 100
            # x_i, y_i = interpolate.splev(np.linspace(0, 1, traj_length / dx * di), spline[0])
            traj = np.array(list(zip(x_i, y_i)))
            d_traj = np.diff(traj, axis=0)
            orient = Geo.angle(d_traj).ravel()

            d_orient = Geo.angle_distance(orient[100:], orient[:-100])

            list_dorient += list(d_orient)
            list_orient += list(orient)

        dtheta = 0.1
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)
        x = (bins[:-1]+bins[1:])/2.
        y, _ = np.histogram(list_orient+list(-np.array(list_orient)), bins, density=True)
        pb.plot(x, y, 'o', c='lightblue')

        y, _ = np.histogram(list_dorient+list(-np.array(list_dorient)), bins, density=True)
        c, s, d, x_fit, y_fit = Fits.centered_gauss_cst_fit(x, y)

        pb.plot(x_fit, y_fit, c='k')
        pb.plot(x, y, 'o', c='grey')

        print(round(s**2, 5), round(np.nanvar(list_dorient), 5), round(np.nanvar(list_orient), 5))
        pb.show()

    def compute_var_orientation_per_nb_carriers(self, redo):
        result_name = 'var_orientation_per_nb_carriers'

        if redo:

            name_n_ant = 'nb_carriers'
            name_attachment = 'attachment_intervals'
            self.exp.load([name_n_ant, name_attachment])

            name_food_x = 'mm10_food_x'
            name_food_y = 'mm10_food_y'
            name_food_xy = 'mm10_food_xy'
            self.exp.load_as_2d(name_food_x, name_food_y, name_food_xy, 'x', 'y')

            init_frame_name = 'first_attachment_time_of_outside_ant'
            self.change_first_frame2d(name_food_xy, init_frame_name)
            self.change_first_frame(name_attachment, init_frame_name)
            self.change_first_frame(name_n_ant, init_frame_name)

            dx = 3
            res_before = [[] for _ in range(20)]
            res_after = [[] for _ in range(20)]

            def do4each_group(df: pd.DataFrame):
                vals = df.reset_index().values
                argsort = np.argsort(vals[:, 2])
                vals = vals[argsort]

                id_exp = vals[0, 0]
                print(id_exp)
                frame1 = vals[0, 2]
                tab_xy = self.exp.get_df(name_food_xy).loc[pd.IndexSlice[id_exp, :frame1], :]
                temp(-1, id_exp, tab_xy)

                for id_exp, id_ant, frame0, dframe in vals:

                    frame1 = frame0+dframe*100
                    tab_xy = self.exp.get_df(name_food_xy).loc[pd.IndexSlice[id_exp, frame0:frame1], :]
                    temp(frame0, id_exp, tab_xy)

            def temp(frame0, id_exp, tab_xy):

                tab_x = tab_xy['x'].values.ravel()
                tab_y = tab_xy['y'].values.ravel()

                mask = np.where(np.abs(np.diff(tab_x)) + np.abs(np.diff(tab_y)) > 0)
                tab_x = tab_x[mask]
                tab_y = tab_y[mask]

                traj_length = int(np.sum(np.sqrt(np.diff(tab_x)**2+np.diff(tab_y)**2)))

                if traj_length > 5:
                    spline = interpolate.splprep([tab_x, tab_y], s=1)
                    x_i, y_i = interpolate.splev(np.linspace(0, 1, traj_length), spline[0])
                    traj_length = np.sum(np.sqrt(np.diff(x_i) ** 2 + np.diff(y_i) ** 2))

                    di = 100
                    x_i, y_i = interpolate.splev(np.linspace(0, 1, traj_length / dx * di), spline[0])
                    traj = np.array(list(zip(x_i, y_i)))
                    d_traj = np.diff(traj, axis=0)

                    orient = Geo.angle(d_traj).ravel()
                    d_orient = Geo.angle_distance(orient[100:], orient[:-100])

                    n_ant = self.exp.get_value(name_n_ant, (id_exp, frame0))
                    if frame0 < 0:
                        res_before[n_ant] += list(d_orient)
                    else:
                        res_after[n_ant] += list(d_orient)

            self.exp.groupby(name_attachment, id_exp_name, do4each_group)

            df_res = pd.DataFrame(index=np.arange(21), columns=['before', 'after'])

            dtheta = 0.1
            bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)
            x = (bins[:-1]+bins[1:])/2.
            for i in range(20):

                if len(res_before[i]) != 0:
                    y, _ = np.histogram(res_before[i], bins, density=True)
                    _, s, _, _, _ = Fits.centered_gauss_cst_fit(x, y)
                    df_res.loc[i, 'before'] = s**2

                if len(res_after[i]) != 0:
                    y, _ = np.histogram(res_after[i], bins, density=True)
                    _, s, _, _, _ = Fits.centered_gauss_cst_fit(x, y)
                    df_res.loc[i, 'after'] = s**2

            self.exp.add_new_dataset_from_df(df=df_res, name=result_name, category=self.category,
                                             label='var_orient over nb of carriers',
                                             description='Variance of the orientation variation'
                                                         ' over the number of carriers before and after the first'
                                                         'attachment time of ant outside ant')
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(self.exp.root, self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(title='', xlabel='nb of carriers', ylabel=r'$\sigma_{orient}^2$')
        ax.set_xlim(0, 20)
        ax.set_xticks(range(0, 21, 2))
        ax.set_ylim(0, 1)
        ax.grid()
        plotter.save(fig)

    def compute_cos_correlation_orientation(self):

        self.exp.load(['mm10_food_x', 'mm10_food_y'])
        init_frame_name = 'first_attachment_time_of_outside_ant'
        self.change_first_frame('mm10_food_x', init_frame_name)
        self.change_first_frame('mm10_food_y', init_frame_name)
        dx = 1.2
        res2 = np.zeros(1000)
        for id_exp in range(1, 61):
            tab_x = self.exp.get_df('mm10_food_x').loc[pd.IndexSlice[id_exp, :0], :].values.ravel()
            tab_y = self.exp.get_df('mm10_food_y').loc[pd.IndexSlice[id_exp, :0], :].values.ravel()

            mask = np.where(np.abs(np.diff(tab_x)) + np.abs(np.diff(tab_y)) > 0)
            tab_x = tab_x[mask]
            tab_y = tab_y[mask]

            spline = interpolate.splprep([tab_x, tab_y], s=1)
            x_i, y_i = interpolate.splev(np.linspace(0, 1, 10000), spline[0])
            traj_length = np.sum(np.sqrt(np.diff(x_i)**2+np.diff(y_i)**2))

            x_i, y_i = interpolate.splev(np.linspace(0, 1, traj_length/dx), spline[0])
            traj = np.array(list(zip(x_i, y_i)))
            d_traj = np.diff(traj, axis=0)
            orient = Geo.angle(d_traj).ravel()

            res = np.zeros(len(orient))
            weight = np.zeros(len(orient))

            for i in range(1, len(orient)):
                res[:-i] += np.cos(Geo.angle_distance(orient[i], orient[i:])).ravel()
                weight[:-i] += 1.

            res /= weight

            res2[:len(res)] += res

        pb.plot(np.arange(0, len(res2)*dx, dx), res2)
        pb.show()

    def compute_food_orientation_diff(self, redo=False, redo_hist=False):

        velocity_phi_name = 'mm1s_food_velocity_phi'
        result_name = 'food_orientation_diff'

        hist_name = result_name+'_hist'
        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        label = 'Food orientation speed (rad/s)'
        hist_label = 'Histogram of the %s' % label
        description = 'Speed of the angular coordinate of the food velocity' \
                      ' (rad, in the food system)'
        hist_description = 'Histogram of the %s' % description
        if redo:
            self.exp.load([velocity_phi_name, 'fps'])

            self.exp.add_copy1d(
                name_to_copy=velocity_phi_name, copy_name=result_name, category=self.category,
                label=label, description=description)

            dt = 2

            def get_velocity4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                fps = self.exp.get_value('fps', id_exp)
                dframe = dt*fps
                dframe2 = int(dframe/2)

                dx = np.array(df.loc[id_exp, :]).ravel()
                dx2 = dx.copy()
                dx[:] = np.nan
                dx[dframe2:-dframe2] = Geo.angle_distance(dx2[:-dframe], dx2[dframe:])

                self.exp.get_df(result_name).loc[id_exp, :] = np.c_[np.abs(np.around(dx, 6))]

            self.exp.groupby(velocity_phi_name, id_exp_name, get_velocity4each_group)

            self.exp.write(result_name)

        self.compute_hist(hist_name=hist_name, name=result_name, bins=bins,
                          hist_label=hist_label, hist_description=hist_description, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel=r'$\dot\varphi$', ylabel='PDF', normed=True)
        # ax.set_ylim(0, 0.7)
        plotter.save(fig)
