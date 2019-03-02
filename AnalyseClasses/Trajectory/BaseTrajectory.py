import numpy as np

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Tools.MiscellaneousTools.Geometry import pts2vect, angle, distance, norm_angle_tab
from math import pi


class AnalyseTrajectory:
    def __init__(self, root, group):
        self.exp = ExperimentGroupBuilder(root).build(group)

    def initialize_xy_orientation_food(self, dynamic_food=False):
        print('x, y')
        id_exp_list = self.exp.set_id_exp_list()

        self.__load_xy_reorientation(dynamic_food=dynamic_food)
        self.__copy_xy0_to_xy(dynamic_food=dynamic_food)
        # self.__translate_xy(dynamic_food=dynamic_food)
        self.__centered_xy_on_food(dynamic_food=dynamic_food)
        self.__convert_xy_to_mm(dynamic_food=dynamic_food)
        self.__orient_all_in_same_direction(id_exp_list, dynamic_food=dynamic_food)
        self.__write_initialize_xy_orientation(dynamic_food)

    def __load_xy_reorientation(self, dynamic_food):
        if dynamic_food is True:
            self.exp.load(['food_x0', 'food_y0'])
        self.exp.load([
            'x0', 'y0', 'absoluteOrientation', 'entrance1', 'entrance2', 'food_center', 'mm2px', 'traj_translation'])

    def __copy_xy0_to_xy(self, dynamic_food):
        print('coping xy0, absoluteOrientation and food0')
        self.exp.add_copy1d(
            name_to_copy='x0', copy_name='x', category='Trajectory',
            label='x (mm)', description='x coordinate (mm, in the initial food system)'
        )
        self.exp.add_copy1d(
            name_to_copy='y0', copy_name='y', category='Trajectory',
            label='y (mm)', description='y coordinate (mm, in the initial food system)'
        )
        self.exp.add_copy1d(
            name_to_copy='absoluteOrientation', copy_name='orientation', category='Trajectory',
            label='orientation (rad)', description='ant orientation (in the initial food system)'
        )
        self.exp.operation('orientation', lambda z: round(z, 3))
        if dynamic_food is True:
            self.exp.add_copy1d(
                name_to_copy='food_x0', copy_name='food_x', category='FoodBase',
                label='x (mm)', description='x coordinate of the food (mm, in the initial food system)'
            )
            self.exp.add_copy1d(
                name_to_copy='food_y0', copy_name='food_y', category='FoodBase',
                label='y (mm)', description='y coordinate of the food (mm, in the initial food system)'
            )

    def __centered_xy_on_food(self, dynamic_food):
        print('centering')
        self.exp.operation_between_2names('x', 'food_center', lambda x, y: x - y, 'x')
        self.exp.operation_between_2names('y', 'food_center', lambda x, y: x - y, 'y')
        if dynamic_food is True:
            self.exp.operation_between_2names('food_x', 'food_center', lambda x, y: x - y, 'x')
            self.exp.operation_between_2names('food_y', 'food_center', lambda x, y: x - y, 'y')

    def __translate_xy(self, dynamic_food):
        print('translating')
        self.exp.operation_between_2names('x', 'traj_translation', lambda x, y: x + y, 'x')
        self.exp.operation_between_2names('y', 'traj_translation', lambda x, y: x + y, 'y')
        if dynamic_food is True:
            self.exp.operation_between_2names('food_x', 'traj_translation', lambda x, y: x + y, 'x')
            self.exp.operation_between_2names('food_y', 'traj_translation', lambda x, y: x + y, 'y')

    def __convert_xy_to_mm(self, dynamic_food):
        print('converting to mm')
        self.exp.operation_between_2names('x', 'mm2px', lambda x, y: round(x / y, 3))
        self.exp.operation_between_2names('y', 'mm2px', lambda x, y: round(x / y, 3))
        if dynamic_food is True:
            self.exp.operation_between_2names('food_x', 'mm2px', lambda x, y: round(x / y, 3))
            self.exp.operation_between_2names('food_y', 'mm2px', lambda x, y: round(x / y, 3))

    def __orient_all_in_same_direction(self, id_exp_list, dynamic_food):
        print('orientation in same direction')
        self.exp.add_copy1d(
            name_to_copy='mm2px', copy_name='traj_reoriented',
            category='Trajectory', label='trajectory reoriented',
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
        self.exp.write(['traj_reoriented', 'orientation', 'x', 'y'])

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
            category='Trajectory',
            label='r',
            description='radial coordinate (in the food system)'
        )
        self.exp.add_copy1d(
            name_to_copy='x', copy_name='phi',
            category='Trajectory',
            label='phi',
            description='angular coordinate (in the food system)'
        )

    def __merge_xy_in2d(self):
        self.exp.add_2d_from_1ds(
            name1='x', name2='y',
            result_name='xy', xname='x', yname='y',
            category='Trajectory', label='coordinates', xlabel='x', ylabel='y',
            description='coordinates of ant positions'
        )

    def compute_speed(self):
        name = 'speed'
        self.exp.load(['x', 'y', 'fps'])
        self.exp.load_timeseries_exp_ant_frame_index()

        self.exp.add_copy1d(
            name_to_copy='x', copy_name=name, category='Trajectory', label='Speed',
            description='Instantaneous speed of the ants'
        )
        self.exp.add_copy1d(
            name_to_copy='x', copy_name=name+'_x', category='Trajectory', label='X speed',
            description='X coordinate of the instantaneous speed of the ants'
        )
        self.exp.add_copy1d(
            name_to_copy='x', copy_name=name+'_y', category='Trajectory', label='Y speed',
            description='Y coordinate of the instantaneous speed of the ants'
        )

        for id_exp in self.exp.timeseries_exp_ant_frame_index:
            for id_ant in self.exp.timeseries_exp_ant_frame_index[id_exp]:
                print(id_exp, id_ant)

                dx = np.array(self.exp.x.df.loc[id_exp, id_ant, :])
                dx1 = dx[1, :].copy()
                dx2 = dx[-2, :].copy()
                dx[1:-1, :] = (dx[2:, :]-dx[:-2, :])/2.
                dx[0, :] = dx1-dx[0, :]
                dx[-1, :] = dx[-1, :]-dx2

                dy = np.array(self.exp.y.df.loc[id_exp, id_ant, :])
                dy1 = dy[1, :].copy()
                dy2 = dy[-2, :].copy()
                dy[1:-1, :] = (dy[2:, :]-dy[:-2, :])/2.
                dy[0, :] = dy1-dy[0, :]
                dy[-1, :] = dy[-1, :]-dy2

                dt = np.array(self.exp.timeseries_exp_ant_frame_index[id_exp][id_ant], dtype=float)
                dt.sort()
                dt[1:-1] = dt[2:]-dt[:-2]
                dt[0] = 1
                dt[-1] = 1
                dx[dt > 2] = np.nan
                dy[dt > 2] = np.nan

                self.exp.speed_x.df.loc[id_exp, id_ant, :] = np.around(dx*self.exp.fps.df.loc[id_exp].fps)
                self.exp.speed_y.df.loc[id_exp, id_ant, :] = np.around(dy*self.exp.fps.df.loc[id_exp].fps)
                self.exp.speed.df.loc[id_exp, id_ant, :] =\
                    np.around(np.sqrt(dx**2+dy**2)*self.exp.fps.df.loc[id_exp].fps, 3)

        self.exp.write([name, name+'_x', name+'_y'])

    def compute_mm10_speed(self):
        name = 'speed'
        category = 'SpeedMM'
        time_window = 10

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=category
        )

        self.exp.write(result_name)

    def compute_mm20_speed(self):
        name = 'speed'
        category = 'SpeedMM'
        time_window = 20

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=category
        )

        self.exp.write(result_name)

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
