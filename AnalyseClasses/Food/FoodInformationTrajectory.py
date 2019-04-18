import numpy as np
import pandas as pd

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_frame_name
from Tools.MiscellaneousTools.ArrayManipulation import get_index_interval_containing
from Tools.Plotter.Plotter import Plotter


class AnalyseFoodInformationTrajectory(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'FoodInformationTrajectory'
        self.exp.data_manager.create_new_category(self.category)

    def compute_w10s_food_direction_error_vs_path_efficiency_indiv(self):
        time = 10
        confidence_name = 'w'+str(time)+'s_food_path_efficiency'
        veracity_name = 'mm'+str(time)+'s_food_direction_error'
        result_name = 'w'+str(time)+'s_food_direction_error_vs_path_efficiency'

        xlabel = 'Food path efficiency'
        ylabel = 'Direction error'

        self.__compute_indiv_info_traj(result_name, confidence_name, veracity_name, xlabel, ylabel)

    def compute_w30s_food_direction_error_vs_path_efficiency_indiv(self):
        time = 30
        confidence_name = 'w'+str(time)+'s_food_path_efficiency'
        veracity_name = 'mm'+str(time)+'s_food_direction_error'
        result_name = 'w'+str(time)+'s_food_direction_error_vs_path_efficiency'

        xlabel = 'Food path efficiency'
        ylabel = 'Direction error'

        self.__compute_indiv_info_traj(result_name, confidence_name, veracity_name, xlabel, ylabel)

    def __compute_indiv_info_traj(self, result_name, confidence_name, veracity_name, xlabel, ylabel):
        self.exp.load([confidence_name, veracity_name])

        def plot2d(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            print(id_exp)

            confidence_df = df.loc[id_exp, :]
            error_df = self.exp.get_df(veracity_name).loc[id_exp, :]
            error_df = error_df.abs()/np.pi

            df_to_plot = confidence_df.join(error_df)
            df_to_plot = df_to_plot.set_index(confidence_name)
            df_to_plot.dropna(inplace=True)

            self.exp.add_new_dataset_from_df(df=df_to_plot, name='to_plot', category=self.category, replace=True)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('to_plot'))
            fig, ax = plotter.plot(xlabel=xlabel, ylabel=ylabel, marker='', title=id_exp)

            ax.set_xlim((0, 1))
            ax.set_ylim((0, 1))
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.set_xticks(np.arange(0, 1.1, 0.2))
            ax.axis('equal')
            ax.grid()
            plotter.save(fig, name=id_exp, sub_folder=result_name)

        self.exp.groupby(confidence_name, id_exp_name, plot2d)

    def w10s_food_direction_error_vs_path_efficiency_velocity(self):
        time = 10
        name_x = 'w'+str(time)+'s_food_path_efficiency'
        name_y = 'mm'+str(time)+'s_food_direction_error'

        result_velocity_x_name = 'w'+str(time)+'s_food_direction_error_vs_path_efficiency_velocity_x'
        result_velocity_y_name = 'w'+str(time)+'s_food_direction_error_vs_path_efficiency_velocity_y'

        self.__compute_velocity(name_x, name_y, result_velocity_x_name, result_velocity_y_name)

    def w30s_food_direction_error_vs_path_efficiency_velocity(self):
        time = 30
        name_x = 'w'+str(time)+'s_food_path_efficiency'
        name_y = 'mm'+str(time)+'s_food_direction_error'

        result_velocity_x_name = 'w'+str(time)+'s_food_direction_error_vs_path_efficiency_velocity_x'
        result_velocity_y_name = 'w'+str(time)+'s_food_direction_error_vs_path_efficiency_velocity_y'

        self.__compute_velocity(name_x, name_y, result_velocity_x_name, result_velocity_y_name)

    def __compute_velocity(self, name_x, name_y, result_velocity_x_name, result_velocity_y_name):
        label_x = 'X of food direction error vs path efficiency velocity'
        label_y = 'Y of food direction error vs path efficiency velocity'
        description_x = 'X coordinates of the velocity of the trajectory' \
                        ' taking food direction error as X and path efficiency velocity as Y'
        description_y = 'Y coordinates of the velocity of the trajectory' \
                        ' taking food direction error as X and path efficiency velocity as Y'
        self.exp.load([name_x, name_y, 'fps'])
        self.exp.add_copy1d(name_to_copy=name_x, copy_name=result_velocity_x_name, category=self.category,
                            label=label_x, description=description_x)
        self.exp.add_copy1d(name_to_copy=name_y, copy_name=result_velocity_y_name, category=self.category,
                            label=label_y, description=description_y)
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

            self.exp.get_df(result_velocity_x_name).loc[id_exp, :] = np.around(dx * fps, 3)
            self.exp.get_df(result_velocity_y_name).loc[id_exp, :] = np.around(dy * fps, 3)
        self.exp.write(result_velocity_x_name)
        self.exp.write(result_velocity_y_name)

    def w10s_food_direction_error_vs_path_efficiency_field(self):
        time = 10
        self.__compute_field(time)

    def w30s_food_direction_error_vs_path_efficiency_field(self):
        time = 30
        self.__compute_field(time)

    def __compute_field(self, time):
        name_confidence = 'w' + str(time) + 's_food_path_efficiency'
        name_veracity = 'mm' + str(time) + 's_food_direction_error'
        name_x = 'w' + str(time) + 's_food_direction_error_vs_path_efficiency_velocity_x'
        name_y = 'w' + str(time) + 's_food_direction_error_vs_path_efficiency_velocity_y'
        self.exp.load([name_x, name_y, name_confidence, name_veracity])
        result_name = 'w' + str(time) + 's_food_direction_error_vs_path_efficiency_field'
        result_name_x = result_name + '_x'
        result_name_y = result_name + '_y'
        label_x = 'X of the vector field food path efficiency vs food direction error'
        label_y = 'Y of the vector field food path efficiency vs food direction error'
        description_x = 'X coordinates of the vector field of the process taking the food path efficiency as X and' \
                        ' food direction error as Y'
        description_y = 'Y coordinates of the vector field of the process taking the food path efficiency as X and' \
                        ' food direction error as Y'
        dc = 0.1
        confidence_intervals = np.arange(0, 1 + dc, dc)
        confidence_intervals2 = np.around((confidence_intervals[1:] + confidence_intervals[:-1]) / 2., 2)
        dv = 0.1
        veracity_intervals = np.arange(0, 1 + dv, dv)
        veracity_intervals2 = np.around((veracity_intervals[1:] + veracity_intervals[:-1]) / 2., 2)
        self.exp.add_new_empty_dataset(name=result_name_x, index_names='confidence', fill_value=0,
                                       column_names=veracity_intervals2, index_values=confidence_intervals2,
                                       category=self.category, label=label_x, description=description_x)
        self.exp.add_new_empty_dataset(name=result_name_y, index_names='confidence', fill_value=0,
                                       column_names=veracity_intervals2, index_values=confidence_intervals2,
                                       category=self.category, label=label_y, description=description_y)
        self.exp.add_new_empty_dataset(name='norm', index_names='confidence', fill_value=0,
                                       column_names=veracity_intervals2, index_values=confidence_intervals2)

        def compute_field4each_group(df):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            frame = df.index.get_level_values(id_frame_name)[0]

            x = float(df.loc[id_exp, frame])
            y = self.exp.get_value(name_y, (id_exp, frame))
            confidence = self.exp.get_value(name_confidence, (id_exp, frame))
            veracity = np.abs(self.exp.get_value(name_veracity, (id_exp, frame)))

            if not np.isnan(confidence) and not np.isnan(veracity) and not np.isnan(x) and not np.isnan(y):
                i_c = min(get_index_interval_containing(confidence, confidence_intervals),
                          len(confidence_intervals2) - 1)
                i_v = min(get_index_interval_containing(veracity, veracity_intervals), len(veracity_intervals2) - 1)

                self.exp.get_df(result_name_x).loc[confidence_intervals2[i_c], veracity_intervals2[i_v]] += x
                self.exp.get_df(result_name_y).loc[confidence_intervals2[i_c], veracity_intervals2[i_v]] += y
                self.exp.get_df('norm').loc[confidence_intervals2[i_c], veracity_intervals2[i_v]] += 1

        self.exp.groupby(name_x, [id_exp_name, id_frame_name], compute_field4each_group)
        self.exp.get_data_object(result_name_x).df /= self.exp.get_df('norm')
        self.exp.get_data_object(result_name_y).df /= self.exp.get_df('norm')
        self.exp.write(result_name_x)
        self.exp.write(result_name_y)
        plotter = Plotter(self.exp.root, self.exp.get_data_object(result_name_x))
        fig, ax = plotter.create_plot()
        mat_x, mat_y = np.meshgrid(confidence_intervals2, veracity_intervals2)
        mat_u = self.exp.get_data_object(result_name_x).get_array()
        mat_v = self.exp.get_data_object(result_name_y).get_array()
        ax.quiver(mat_x, mat_y, mat_u, mat_v)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        plotter.save(fig, name=result_name)
