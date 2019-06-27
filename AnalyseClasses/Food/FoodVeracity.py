import numpy as np
import pandas as pd

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name
from Tools.MiscellaneousTools.Geometry import angle_df, angle_distance
from Tools.Plotter.Plotter import Plotter


class AnalyseFoodVeracity(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'FoodVeracity'

    def compute_food_direction_error(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        food_exit_angle_name = 'food_exit_angle'
        food_phi_name = 'food_velocity_phi'
        result_name = 'food_direction_error'

        dtheta = np.pi/25.
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)

        result_label = 'Food direction error (rad)'
        result_description = 'Angle between the food velocity and the food-exit vector,' \
                             'which gives in radian how much the food is not going in the good direction'
        if redo:
            self.exp.load([food_exit_angle_name, food_phi_name])

            tab = angle_distance(self.exp.get_df(food_exit_angle_name)[food_exit_angle_name],
                                 self.exp.get_df(food_phi_name)[food_phi_name])

            self.exp.add_copy1d(name_to_copy=food_phi_name, copy_name=result_name, category=self.category,
                                label=result_label, description=result_description)

            self.exp.change_values(result_name, np.around(tab, 5))

            self.exp.write(result_name)

        else:
            self.exp.load(result_name)

        self.__plot_indiv(result_name, redo, redo_plot_indiv)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter2 = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig2, ax2 = plotter2.plot(xlabel='Food direction error', ylabel='PDF', normed=True)
        plotter2.save(fig2)

    def compute_food_direction_error_evol(self, redo=False):
        name = 'food_direction_error'
        result_name = name+'_hist_evol'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        dx = 0.25
        start_frame_intervals = np.arange(0, 4., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        func = lambda x: np.abs(x)

        hist_label = 'Food direction error distribution over time (rad)'
        hist_description = 'Histogram of the angle between the food velocity and the food-exit vector,' \
                           'which gives in radian how much the food is not going in the good direction (rad)'

        if redo:
            self.exp.load(name)
            self.exp.operation(name, func)
            self.exp.hist1d_evolution(name_to_hist=name, start_index_intervals=start_frame_intervals,
                                      end_index_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category,
                                      label=hist_label, description=hist_description)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\varphi$', ylabel='PDF', normed=True, label_suffix='s')
        plotter.save(fig)

    def compute_food_direction_error_variance_evol(self, redo=False):
        name = 'food_direction_error'
        result_name = name + '_var_evol'
        init_frame_name = 'food_first_frame'

        dx = 0.25
        start_frame_intervals = np.arange(0, 4., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        label = 'Variance of the food direction error distribution over time'
        description = 'Variance of the angle between the food velocity and the food-exit vector,' \
                      'which gives in radian how much the food is not going in the good direction'

        if redo:
            self.exp.load([name, init_frame_name])

            new_times = 'new_times'
            self.exp.add_copy1d(name_to_copy=name, copy_name=new_times, replace=True)
            self.exp.get_df(new_times).loc[:, new_times] = self.exp.get_index(new_times).get_level_values(id_frame_name)
            self.exp.operation_between_2names(name1=new_times, name2=init_frame_name, func=lambda x, y: x - y)
            self.exp.get_df(new_times).reset_index(inplace=True)

            self.exp.get_df(name).reset_index(inplace=True)
            self.exp.get_df(name).loc[:, id_frame_name] = self.exp.get_df(new_times).loc[:, new_times]
            self.exp.get_df(name).set_index([id_exp_name, id_frame_name], inplace=True)

            self.exp.variance_evolution(name_to_var=name, start_index_intervals=start_frame_intervals,
                                        end_index_intervals=end_frame_intervals,
                                        category=self.category, result_name=result_name,
                                        label=label, description=description)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(yscale='log', xlabel='Time (s)', ylabel='Variance', label_suffix='s', label='Variance')
        plotter.draw_horizontal_line(ax, 1.75)
        plotter.plot_fit(preplot=(fig, ax), window=[0, 11])
        plotter.save(fig)

    def compute_food_direction_error_hist_evol_around_first_attachment(self, redo=False):
        name = 'food_direction_error'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        result_name = name+'_hist_evol_around_first_attachment'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        dx = 0.25
        start_frame_intervals = np.arange(-1, 3., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        result_label = 'Food direction error histogram evolution over time'
        result_description = 'Evolution over time of the histogram of food error direction,negative times (s)' \
                             ' correspond to periods before the first attachment of an outside ant'

        if redo:
            self.exp.load([name, first_attachment_name])

            new_times = 'new_times'
            self.exp.add_copy1d(name_to_copy=name, copy_name=new_times, replace=True)
            self.exp.get_df(new_times).loc[:, new_times] = self.exp.get_index(new_times).get_level_values(id_frame_name)
            self.exp.operation_between_2names(name1=new_times, name2=first_attachment_name, func=lambda x, y: x - y)
            self.exp.get_df(new_times).reset_index(inplace=True)

            self.exp.get_df(name).reset_index(inplace=True)
            self.exp.get_df(name).loc[:, id_frame_name] = self.exp.get_df(new_times).loc[:, new_times]
            self.exp.get_df(name).set_index([id_exp_name, id_frame_name], inplace=True)

            self.exp.operation(name, lambda x: np.abs(x))

            self.exp.hist1d_evolution(name_to_hist=name, start_index_intervals=start_frame_intervals,
                                      end_index_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category,
                                      label=result_label, description=result_description)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\varphi$', ylabel='PDF', normed=True, label_suffix=' s')
        plotter.save(fig)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\varphi$', ylabel='PDF', label_suffix=' s')
        plotter.save(fig, suffix='non_norm')

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(yscale='log', xlabel=r'$\varphi$', ylabel='PDF', normed=True, label_suffix='s')
        plotter.save(fig, suffix='exp')

        self.exp.get_df(result_name).index = self.exp.get_df(result_name).index**2
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(yscale='log', xlabel=r'$\varphi$', ylabel='PDF', normed=True, label_suffix='s')
        plotter.save(fig, suffix='gauss')

    def compute_food_direction_error_variance_evol_around_first_attachment(self, redo=False):
        name = 'food_direction_error'
        result_name = name + '_var_evol_around_first_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.25
        start_frame_intervals = np.arange(-1, 3., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        label = 'Variance of the food direction error distribution over time'
        description = 'Variance of the angle between the food velocity and the food-exit vector,' \
                      'which gives in radian how much the food is not going in the good direction'

        if redo:
            self.exp.load([name, init_frame_name])

            new_times = 'new_times'
            self.exp.add_copy1d(name_to_copy=name, copy_name=new_times, replace=True)
            self.exp.get_df(new_times).loc[:, new_times] = self.exp.get_index(new_times).get_level_values(id_frame_name)
            self.exp.operation_between_2names(name1=new_times, name2=init_frame_name, func=lambda x, y: x - y)
            self.exp.get_df(new_times).reset_index(inplace=True)

            self.exp.get_df(name).reset_index(inplace=True)
            self.exp.get_df(name).loc[:, id_frame_name] = self.exp.get_df(new_times).loc[:, new_times]
            self.exp.get_df(name).set_index([id_exp_name, id_frame_name], inplace=True)

            self.exp.variance_evolution(name_to_var=name, start_index_intervals=start_frame_intervals,
                                        end_index_intervals=end_frame_intervals,
                                        category=self.category, result_name=result_name,
                                        label=label, description=description)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(yscale='log', xlabel='Time (s)', ylabel='Variance', label_suffix='s', label='Variance')
        plotter.draw_vertical_line(ax)
        plotter.draw_horizontal_line(ax, 1.75)
        plotter.plot_fit(preplot=(fig, ax), window=[0, 9])
        plotter.save(fig)

    def compute_mm1s_food_direction_error(self, redo=False, redo_plot_indiv=False, redo_hist=False):

        name = 'food_direction_error'
        result_name = 'mm1s_' + name
        food_exit_angle_name = 'mm1s_food_exit_angle'
        vel_name = 'mm1s_food_velocity'
        time_window = 100
        self.__compute_food_direction_error(result_name, self.category, food_exit_angle_name, vel_name,
                                            time_window, redo, redo_hist, redo_plot_indiv)

    def compute_mm10s_food_direction_error(self, redo=False, redo_plot_indiv=False, redo_hist=False):

        name = 'food_direction_error'
        result_name = 'mm10s_' + name
        food_exit_angle_name = 'mm10s_food_exit_angle'
        vel_name = 'mm10s_food_velocity'
        time_window = 1000
        self.__compute_food_direction_error(result_name, self.category, food_exit_angle_name, vel_name,
                                            time_window, redo, redo_hist, redo_plot_indiv)

    def compute_mm30s_food_direction_error(self, redo=False, redo_plot_indiv=False, redo_hist=False):

        name = 'food_direction_error'
        result_name = 'mm30s_' + name
        food_exit_angle_name = 'mm30s_food_exit_angle'
        vel_name = 'mm30s_food_velocity'
        time_window = 3000
        self.__compute_food_direction_error(result_name, self.category, food_exit_angle_name, vel_name,
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

            tab = angle_distance(self.exp.get_df(food_exit_angle_name)[food_exit_angle_name], vel_phi)

            self.exp.add_copy1d(name_to_copy=vel_name_x, copy_name=result_name, category=category,
                                label=result_label, description=result_description)

            self.exp.change_values(result_name, np.around(tab, 3))

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        self.__plot_indiv(result_name, redo, redo_plot_indiv)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter2 = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig2, ax2 = plotter2.plot(xlabel='Food direction error', ylabel='PDF', normed=True)
        plotter2.save(fig2)

    def __plot_indiv(self, result_name, redo, redo_plot_indiv):
        if redo or redo_plot_indiv:
            attachment_name = 'outside_ant_carrying_intervals'
            self.exp.load(['fps', attachment_name])

            def plot4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                fps = self.exp.get_value('fps', id_exp)
                df2 = df.loc[id_exp, :]
                df2 = df2.abs()
                df2.index = df2.index / fps

                self.exp.add_new_dataset_from_df(df=df2, name='temp', category=self.category, replace=True)

                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
                fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Food direction error', marker='', title=id_exp)

                attachments = self.exp.get_df(attachment_name).loc[id_exp, :]
                attachments.reset_index(inplace=True)
                attachments = np.array(attachments)

                colors = plotter.color_object.create_cmap('hot_r', set(list(attachments[:, 0])))
                for id_ant, frame, inter in attachments:
                    ax.axvline(frame / fps, c=colors[str(id_ant)], alpha=0.5)

                ax.set_ylim((0, np.pi))
                plotter.save(fig, name=id_exp, sub_folder=result_name)

            self.exp.groupby(result_name, id_exp_name, plot4each_group)
