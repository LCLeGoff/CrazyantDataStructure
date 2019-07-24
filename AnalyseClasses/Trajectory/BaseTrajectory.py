import numpy as np
import pandas as pd

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.MiscellaneousTools.Geometry import pts2vect, angle, distance, norm_angle_tab
from math import pi

from Tools.Plotter.Plotter import Plotter


class AnalyseTrajectory(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'Trajectory'

    def interpolate_xy_orientation_food(self, dynamic_food=False):
        print('x, y')

        self.__load_xy0_reorientation(dynamic_food=dynamic_food)
        self.__copy_xy0_to_interpolated_xy0(dynamic_food=dynamic_food)
        self.__interpolate_xy_and_orientation(dynamic_food=dynamic_food)
        self.__write_interpolated_xy0_orientation(dynamic_food=dynamic_food)

    def initialize_xy_orientation_food(self, dynamic_food=False):
        print('x, y')
        id_exp_list = self.exp.set_id_exp_list()

        self.__load_interpolated_xy0_reorientation(dynamic_food=dynamic_food)
        self.__copy_interpolated_xy0_to_xy(dynamic_food=dynamic_food)
        self.__centered_xy_on_food(dynamic_food=dynamic_food)
        self.__convert_xy_to_mm(dynamic_food=dynamic_food)
        self.__orient_all_in_same_direction(id_exp_list, dynamic_food=dynamic_food)
        self.__write_initialize_xy_orientation(dynamic_food)

    def __load_xy0_reorientation(self, dynamic_food):
        if dynamic_food is True:
            self.exp.load(['food_x0', 'food_y0'])

        if  self.exp.is_name_existing('decrossed_x0'):
            self.exp.load_as_2d('decrossed_x0', 'decrossed_y0', 'xy', 'x', 'y')
        else:
            self.exp.load_as_2d('x0', 'y0', 'xy', 'x', 'y')
        self.exp.load([
            'x0', 'y0', 'absoluteOrientation'])

    def __copy_xy0_to_interpolated_xy0(self, dynamic_food):
        print('coping xy0, absoluteOrientation and food0')

        self.exp.add_copy1d(
            name_to_copy='x0', copy_name='interpolated_x0', category=self.category,
            label='x (px)', description='x coordinate, linearly interpolated (px, in the camera system)'
        )
        self.exp.add_copy1d(
            name_to_copy='y0', copy_name='interpolated_y0', category=self.category,
            label='y (px)', description='y coordinate, linearly interpolated (px, in the camera system)'
        )

        self.exp.add_copy1d(
            name_to_copy='absoluteOrientation', copy_name='interpolatedAbsoluteOrientation', category=self.category,
            label='orientation (rad)', description='ant orientation, linearly interpolated (rad, in the camera system)'
        )

        if dynamic_food is True:
            self.exp.add_copy1d(
                name_to_copy='food_x0', copy_name='interpolated_food_x0', category='FoodBase',
                label='x (px)', description='x coordinate of the food, linearly interpolated (px, in the camera system)'
            )
            self.exp.add_copy1d(
                name_to_copy='food_y0', copy_name='interpolated_food_y0', category='FoodBase',
                label='y (px)', description='y coordinate of the food, linearly interpolated (px, in the camera system)'
            )

    def __interpolate_xy_and_orientation(self, dynamic_food):
        print('interpolating interpolated xy0, absoluteOrientation and food0')

        self.focused_name = 'interpolated_x0'
        res_df = self.exp.groupby(self.focused_name, [id_exp_name, id_ant_name], self.__interpolate_time_series1d)
        self.exp.change_df(self.focused_name, res_df)

        self.focused_name = 'interpolated_y0'
        res_df = self.exp.groupby(self.focused_name, [id_exp_name, id_ant_name], self.__interpolate_time_series1d)
        self.exp.change_df(self.focused_name, res_df)

        self.focused_name = 'interpolatedAbsoluteOrientation'
        res_df = self.exp.groupby(self.focused_name, [id_exp_name, id_ant_name], self.__interpolate_time_series1d)
        self.exp.change_df(self.focused_name, res_df)

        if dynamic_food:
            self.focused_name = 'interpolated_food_x0'
            res_df = self.exp.groupby(self.focused_name, id_exp_name, self.__interpolate_time_series1d)
            self.exp.change_df(self.focused_name, res_df)

            self.focused_name = 'interpolated_food_y0'
            res_df = self.exp.groupby(self.focused_name, id_exp_name, self.__interpolate_time_series1d)
            self.exp.change_df(self.focused_name, res_df)

    @staticmethod
    def __interpolate_time_series1d(df: pd.DataFrame):

        nan_locations = np.where(np.isnan(df))[0]
        if len(nan_locations) != 0:
            prev_nan_loc = nan_locations[0]
            for i in range(0, len(nan_locations)-1):
                current_nan_loc = nan_locations[i]
                if nan_locations[i+1] - current_nan_loc > 1:

                    val0 = float(df.iloc[prev_nan_loc-1])
                    val1 = float(df.iloc[current_nan_loc+1])
                    num = current_nan_loc-prev_nan_loc+1
                    val_range = np.c_[np.linspace(val0, val1, num+1, endpoint=False)[1:]]

                    df.iloc[prev_nan_loc:current_nan_loc+1] = np.around(val_range, 2)

                    prev_nan_loc = nan_locations[i+1]

            current_nan_loc = nan_locations[-1]
            val0 = float(df.iloc[prev_nan_loc-1])
            val1 = float(df.iloc[current_nan_loc+1])
            num = current_nan_loc-prev_nan_loc+1
            val_range = np.c_[np.linspace(val0, val1, num+1, endpoint=False)[1:]]
            df.iloc[prev_nan_loc:current_nan_loc+1] = np.around(val_range, 2)

        return df

    def __write_interpolated_xy0_orientation(self, dynamic_food):
        if dynamic_food is True:
            self.exp.write(['interpolated_food_x0', 'interpolated_food_y0'])
        self.exp.write(['interpolatedAbsoluteOrientation', 'interpolated_x0', 'interpolated_y0'])

    def __load_interpolated_xy0_reorientation(self, dynamic_food):
        if dynamic_food is True:
            self.exp.load(['interpolated_food_x0', 'interpolated_food_y0'])
        self.exp.load([
            'interpolated_x0', 'interpolated_y0', 'interpolatedAbsoluteOrientation',
            'entrance1', 'entrance2', 'exit0_1', 'exit0_2',
            'food_center', 'mm2px'])

    def __copy_interpolated_xy0_to_xy(self, dynamic_food):
        print('coping xy0, absoluteOrientation and food0')
        self.exp.add_copy2d(
            name_to_copy='exit0_1', copy_name='exit1', category=self.category,
            label='Exit position 1 (setup system)',
            xlabel='x coordinates', ylabel='y coordinates',
            description='Coordinates of one of the points defining the exit in the setup system')
        self.exp.add_copy2d(
            name_to_copy='exit0_2', copy_name='exit2', category=self.category,
            label='Exit position 2 (setup system)',
            xlabel='x coordinates', ylabel='y coordinates',
            description='Coordinates of one of the points defining the exit in the setup system')

        self.exp.add_copy1d(
            name_to_copy='interpolated_x0', copy_name='x', category=self.category,
            label='x (mm)', description='x coordinate (mm, in the initial food system)'
        )
        self.exp.add_copy1d(
            name_to_copy='interpolated_y0', copy_name='y', category=self.category,
            label='y (mm)', description='y coordinate (mm, in the initial food system)'
        )

        self.exp.add_copy1d(
            name_to_copy='interpolatedAbsoluteOrientation', copy_name='orientation', category=self.category,
            label='orientation (rad)', description='ant orientation (in the initial food system)'
        )

        if dynamic_food is True:
            self.exp.add_copy1d(
                name_to_copy='interpolated_food_x0', copy_name='food_x', category='FoodBase',
                label='x (mm)', description='x coordinate of the food (mm, in the initial food system)'
            )
            self.exp.add_copy1d(
                name_to_copy='interpolated_food_y0', copy_name='food_y', category='FoodBase',
                label='y (mm)', description='y coordinate of the food (mm, in the initial food system)'
            )

    def __centered_xy_on_food(self, dynamic_food):
        print('centering')
        self.exp.operation_between_2names('exit1', 'food_center', lambda x, y: x - y, 'x', 'x')
        self.exp.operation_between_2names('exit1', 'food_center', lambda x, y: x - y, 'y', 'y')
        self.exp.operation_between_2names('exit2', 'food_center', lambda x, y: x - y, 'x', 'x')
        self.exp.operation_between_2names('exit2', 'food_center', lambda x, y: x - y, 'y', 'y')

        self.exp.operation_between_2names('x', 'food_center', lambda x, y: x - y, 'x')
        self.exp.operation_between_2names('y', 'food_center', lambda x, y: x - y, 'y')
        if dynamic_food is True:
            self.exp.operation_between_2names('food_x', 'food_center', lambda x, y: x - y, 'x')
            self.exp.operation_between_2names('food_y', 'food_center', lambda x, y: x - y, 'y')

    def __convert_xy_to_mm(self, dynamic_food):
        print('converting to mm')
        self.exp.operation_between_2names('exit1', 'mm2px', lambda x, y: round(x / y, 3), 'x', 'x')
        self.exp.operation_between_2names('exit1', 'mm2px', lambda x, y: round(x / y, 3), 'y', 'y')
        self.exp.operation_between_2names('exit2', 'mm2px', lambda x, y: round(x / y, 3), 'x', 'x')
        self.exp.operation_between_2names('exit2', 'mm2px', lambda x, y: round(x / y, 3), 'y', 'y')

        self.exp.operation_between_2names('x', 'mm2px', lambda x, y: round(x / y, 3))
        self.exp.operation_between_2names('y', 'mm2px', lambda x, y: round(x / y, 3))
        if dynamic_food is True:
            self.exp.operation_between_2names('food_x', 'mm2px', lambda x, y: round(x / y, 3))
            self.exp.operation_between_2names('food_y', 'mm2px', lambda x, y: round(x / y, 3))

    def __orient_all_in_same_direction(self, id_exp_list, dynamic_food):
        print('orientation in same direction')
        self.exp.add_copy1d(
            name_to_copy='mm2px', copy_name='traj_reoriented',
            category=self.category, label='trajectory reoriented',
            description='Trajectories reoriented to be in the same orientation of the other experiments'
        )
        for id_exp in id_exp_list:
            food_center = np.array(self.exp.food_center.get_row(id_exp))
            entrance_pts1 = np.array(self.exp.entrance1.get_row(id_exp))
            entrance_pts2 = np.array(self.exp.entrance1.get_row(id_exp))

            entrance_vector = pts2vect(entrance_pts1, entrance_pts2)
            entrance_angle = angle([1, 0], entrance_vector)
            entrance_pts1_centered = entrance_pts1 - food_center

            self.__orient_in_same_direction(entrance_angle, entrance_pts1_centered, id_exp, dynamic_food)

    def __orient_in_same_direction(self, entrance_angle, entrance_pts1_centered, id_exp, dynamic_food):
        is_setup_horizontal = abs(entrance_angle) < pi / 4
        if is_setup_horizontal:
            self.__orient_in_same_horizontal_direction(entrance_pts1_centered, id_exp, dynamic_food)
        else:
            self.__orient_in_same_vertical_direction(entrance_pts1_centered, id_exp, dynamic_food)

    def __orient_in_same_vertical_direction(self, entrance_pts1_centered, id_exp, dynamic_food):
        experiment_orientation = angle([0, 1], entrance_pts1_centered)
        self.__invert_if_not_good_orientation(experiment_orientation, id_exp, dynamic_food)

    def __orient_in_same_horizontal_direction(self, entrance_pts1_centered, id_exp, dynamic_food):
        experiment_orientation = angle([1, 0], entrance_pts1_centered)
        self.__invert_if_not_good_orientation(experiment_orientation, id_exp, dynamic_food)

    def __invert_if_not_good_orientation(self, a, id_exp, dynamic_food):
        if abs(a) > pi / 2:
            self.exp.exit1.operation_on_id_exp(id_exp, lambda z: z * -1)
            self.exp.exit2.operation_on_id_exp(id_exp, lambda z: z * -1)

            self.exp.x.operation_on_id_exp(id_exp, lambda z: z * -1)
            self.exp.y.operation_on_id_exp(id_exp, lambda z: z * -1)
            self.exp.orientation.operation_on_id_exp(id_exp, lambda z: norm_angle_tab(z+np.pi))
            self.exp.traj_reoriented.df.loc[id_exp] = 1
            if dynamic_food is True:
                self.exp.food_x.operation_on_id_exp(id_exp, lambda z: z * -1)
                self.exp.food_y.operation_on_id_exp(id_exp, lambda z: z * -1)
        else:
            self.exp.traj_reoriented.df.loc[id_exp] = 0

    def __write_initialize_xy_orientation(self, dynamic_food):
        if dynamic_food is True:
            self.exp.write(['food_x', 'food_y'])
        self.exp.write(['exit1', 'exit2', 'traj_reoriented', 'orientation', 'x', 'y'])

    def compute_r_phi(self):
        print('r, phi')
        self.exp.load(['x', 'y'])

        self.__merge_xy_in2d()
        self.__copy_xy_to_r_phi()

        r = np.around(distance([0, 0], self.exp.xy.get_array()), 3)
        phi = np.around(angle([1, 0], self.exp.xy.get_array()), 3)
        self.exp.r.replace_values(r)
        self.exp.phi.replace_values(phi)

        self.exp.write(['r', 'phi'])

    def __copy_xy_to_r_phi(self):
        self.exp.add_copy1d(
            name_to_copy='x', copy_name='r',
            category=self.category,
            label='r',
            description='radial coordinate (in the food system)'
        )
        self.exp.add_copy1d(
            name_to_copy='x', copy_name='phi',
            category=self.category,
            label='phi',
            description='angular coordinate (in the food system)'
        )

    def __merge_xy_in2d(self):
        self.exp.add_2d_from_1ds(
            name1='x', name2='y',
            result_name='xy', xname='x', yname='y',
            category=self.category, label='coordinates', xlabel='x', ylabel='y',
            description='coordinates of ant positions'
        )

    def compute_mm10_traj(self):
        name_x = 'x'
        name_y = 'y'
        self.exp.load([name_x, name_y])
        time_window = 10

        category = 'TrajMM'
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(name_to_average=name_x, time_window=time_window,
                                                                    category=category)
        self.exp.write(result_name)

        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(name_to_average=name_y, time_window=time_window,
                                                                    category=category)
        self.exp.write(result_name)

    def compute_speed(self, redo=False, redo_hist=False):
        name = 'speed'
        hist_name = 'speed_hist'
        bins = np.arange(0, 8e2, 1)
        hist_label = 'Distribution of the speed (mm/s)'
        hist_description = 'Distribution of the instantaneous speed of the ants (mm/s)'

        if redo:
            name_x = 'mm10_x'
            name_y = 'mm10_y'
            self.exp.load_as_2d(name_x, name_y, result_name='xy', xname='x', yname='y', replace=True)
            self.exp.load('fps')

            self.exp.add_copy1d(
                name_to_copy=name_x, copy_name=name, category=self.category, label='Speed',
                description='Instantaneous speed of the ants'
            )
            self.exp.add_copy1d(
                name_to_copy=name_x, copy_name=name + '_x', category=self.category, label='X speed',
                description='X coordinate of the instantaneous speed of the ants'
            )
            self.exp.add_copy1d(
                name_to_copy=name_x, copy_name=name + '_y', category=self.category, label='Y speed',
                description='Y coordinate of the instantaneous speed of the ants'
            )

            def compute_speed4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                id_ant = df.index.get_level_values(id_ant_name)[0]
                frames = df.index.get_level_values(id_frame_name)
                print(id_exp, id_ant)

                dx = np.array(df.x)
                dx1 = dx[1].copy()
                dx2 = dx[-2].copy()
                dx[1:-1] = (dx[2:]-dx[:-2])/2.
                dx[0] = dx1-dx[0]
                dx[-1] = dx[-1]-dx2

                dy = np.array(df.y)
                dy1 = dy[1].copy()
                dy2 = dy[-2].copy()
                dy[1:-1] = (dy[2:]-dy[:-2])/2.
                dy[0] = dy1-dy[0]
                dy[-1] = dy[-1]-dy2

                dt = np.array(frames, dtype=float)
                dt.sort()
                dt[1:-1] = dt[2:]-dt[:-2]
                dt[0] = 1
                dt[-1] = 1
                dx[dt > 2] = np.nan
                dy[dt > 2] = np.nan

                fps = self.exp.get_value('fps', id_exp)
                self.exp.speed_x.df.loc[id_exp, id_ant, :] = np.c_[np.around(dx * fps, 3)]
                self.exp.speed_y.df.loc[id_exp, id_ant, :] = np.c_[np.around(dy * fps, 3)]
                self.exp.speed.df.loc[id_exp, id_ant, :] = np.c_[np.around(np.sqrt(dx**2+dy**2) * fps, 3)]

            self.exp.groupby('xy', [id_exp_name, id_ant_name], compute_speed4each_group)

            self.exp.write([name, name+'_x', name+'_y'])

        if redo or redo_hist:
            self.exp.load(name)
            self.exp.hist1d(name_to_hist=name, result_name=hist_name,
                            bins=bins, label=hist_label, description=hist_description)
            self.exp.write(hist_name)

        else:
            self.exp.load(hist_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(yscale='log', normed=True)
        plotter.save(fig)

    def compute_mm10_speed(self):
        category = 'SpeedMM'
        time_window = 10

        name = 'speed'
        name_x = 'speed_x'
        name_y = 'speed_y'
        names = [name, name_x, name_y]

        self.exp.load(names)
        for n in names:
            result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(name_to_average=n, time_window=time_window,
                                                                        category=category)
            self.exp.write(result_name)

    def compute_mm20_speed(self, redo=False, redo_hist=False):
        name = 'speed'
        category = 'SpeedMM'
        time_window = 20
        result_name = 'mm' + str(time_window) + '_' + name
        hist_name = result_name + '_hist'
        bins = np.arange(0, 500, 1)
        hist_label = 'Distribution of the speed (mm/s), MM '+str(time_window)
        hist_description = 'Distribution of the instantaneous speed of the ants (mm/s)' \
                           ' smoothed with a moving mean of window length ' + str(time_window) + ' frames'
        if redo:
            self.exp.load(name)
            result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
                name_to_average=name, time_window=time_window, category=category
            )
            self.exp.write(result_name)

        self.compute_hist(hist_name=hist_name, name=result_name, bins=bins,
                          hist_label=hist_label, hist_description=hist_description, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(yscale='log', normed=True)
        plotter.save(fig)

    def compute_mm1s_speed(self, redo=False, redo_hist=False):
        name = 'speed'
        category = 'SpeedMM'
        time_window = 100
        result_name = 'mm1s_' + name
        hist_name = result_name + '_hist'
        bins = np.arange(0, 500, 1)
        hist_label = 'Distribution of the speed (mm/s), MM 1s'
        hist_description = 'Distribution of the instantaneous speed of the ants (mm/s)' \
                           ' smoothed with a moving mean of window length 1 second'
        if redo:
            self.exp.load(name)
            self.exp.moving_mean4exp_ant_frame_indexed_1d(
                name_to_average=name, time_window=time_window, result_name=result_name, category=category
            )
            self.exp.write(result_name)

        self.compute_hist(hist_name=hist_name, name=result_name, bins=bins,
                          hist_label=hist_label, hist_description=hist_description, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(yscale='log', normed=True)
        plotter.save(fig)

    def compute_mm10_orientation(self):
        name = 'orientation'
        category = 'OrientationMM'
        time_window = 10

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=category
        )

        self.exp.write(result_name)

    def compute_mm20_orientation(self):
        name = 'orientation'
        category = 'OrientationMM'
        time_window = 20

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=category
        )

        self.exp.write(result_name)
