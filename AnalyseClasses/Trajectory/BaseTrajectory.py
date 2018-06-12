import numpy as np

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Tools.MiscellaneousTools.Geometry import pts2vect, angle, distance
from math import pi


class AnalyseTrajectory:
    def __init__(self, root, group):
        self.exp = ExperimentGroupBuilder(root).build(group)

    def compute_x_y(self, id_exp_list=None):
        print('x, y')
        id_exp_list = self.exp.set_id_exp_list(id_exp_list)

        self.exp.load(['x0', 'y0', 'entrance1', 'entrance2', 'food_center', 'mm2px', 'traj_translation'])
        self.copy_xy0_to_xy()

        self.centered_xy_on_food()
        self.translate_xy()
        self.convert_xy_to_mm()
        self.orient_all_xy_in_same_direction(id_exp_list)

        self.exp.write(['x', 'y'])

    def orient_all_xy_in_same_direction(self, id_exp_list):
        for id_exp in id_exp_list:
            food_center = np.array(self.exp.food_center.get_row(id_exp))
            entrance_pts1 = np.array(self.exp.entrance1.get_row(id_exp))
            entrance_pts2 = np.array(self.exp.entrance1.get_row(id_exp))

            entrance_vector = pts2vect(entrance_pts1, entrance_pts2)
            entrance_angle = angle([1, 0], entrance_vector)
            entrance_pts1_centered = entrance_pts1 - food_center

            self.orient_xy_in_same_direction(entrance_angle, entrance_pts1_centered, id_exp)

    def orient_xy_in_same_direction(self, entrance_angle, entrance_pts1_centered, id_exp):
        is_setup_horizontal = abs(entrance_angle) < pi / 4
        if is_setup_horizontal:
            self.orient_xy_in_same_horizontal_direction(entrance_pts1_centered, id_exp)
        else:
            self.orient_xy_in_same_vertical_direction(entrance_pts1_centered, id_exp)

    def orient_xy_in_same_vertical_direction(self, entrance_pts1_centered, id_exp):
        experiment_orientation = angle([0, 1], entrance_pts1_centered)
        self.invert_if_not_good_orientation(experiment_orientation, id_exp)

    def orient_xy_in_same_horizontal_direction(self, entrance_pts1_centered, id_exp):
        experiment_orientation = angle([1, 0], entrance_pts1_centered)
        self.invert_if_not_good_orientation(experiment_orientation, id_exp)

    def inverse_xy_orientation(self, id_exp):
        self.exp.x.operation_on_id_exp(id_exp, lambda z: z * -1)
        self.exp.y.operation_on_id_exp(id_exp, lambda z: z * -1)

    def invert_if_not_good_orientation(self, a, id_exp):
        if abs(a) > pi / 2:
            self.inverse_xy_orientation(id_exp)

    def copy_xy0_to_xy(self):
        self.exp.add_copy1d(
            name_to_copy='x0', copy_name='x', category='Trajectory',
            label='x', description='x coordinate (in the food system)'
        )
        self.exp.add_copy1d(
            name_to_copy='y0', copy_name='y', category='Trajectory',
            label='y', description='y coordinate (in the food system)'
        )

    def convert_xy_to_mm(self):
        self.exp.operation_between_2names('x', 'mm2px', lambda x, y: round(x / y, 2))
        self.exp.operation_between_2names('y', 'mm2px', lambda x, y: round(x / y, 2))

    def translate_xy(self):
        self.exp.operation_between_2names('x', 'traj_translation', lambda x, y: x + y, 'x')
        self.exp.operation_between_2names('y', 'traj_translation', lambda x, y: x + y, 'y')

    def centered_xy_on_food(self):
        self.exp.operation_between_2names('x', 'food_center', lambda x, y: x - y, 'x')
        self.exp.operation_between_2names('y', 'food_center', lambda x, y: x - y, 'y')

    def compute_r_phi(self):
        print('r, phi')
        self.exp.load(['x', 'y'])

        self.merge_xy_in2d()
        self.copy_xy_to_r_phi()

        r = np.around(distance([0, 0], self.exp.xy.get_array()), 3)
        phi = np.around(angle([1, 0], self.exp.xy.get_array()), 3)
        self.exp.r.replace_values(r)
        self.exp.phi.replace_values(phi)

        self.exp.write(['r', 'phi'])

    def copy_xy_to_r_phi(self):
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

    def merge_xy_in2d(self):
        self.exp.add_2d_from_1ds(
            name1='x', name2='y',
            result_name='xy', xname='x', yname='y',
            category='Trajectory', label='coordinates', xlabel='x', ylabel='y',
            description='coordinates of ant positions'
        )
