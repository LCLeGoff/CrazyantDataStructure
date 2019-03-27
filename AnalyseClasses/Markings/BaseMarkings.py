import numpy as np

from pandas import IndexSlice as IdxSc

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.MiscellaneousTools.Geometry import distance
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class AnalyseMarkings(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.pd_idx_manager = PandasIndexManager()

    def compute_xy_marking(self):
        name = 'xy_markings'
        print(name)
        self.exp.load(['x', 'y', 'markings'])

        self.exp.filter_with_time_occurrences(name_to_filter='x', filter_name='markings', result_name='x_markings')
        self.exp.filter_with_time_occurrences(name_to_filter='y', filter_name='markings', result_name='y_markings')

        self.exp.add_2d_from_1ds(
            name1='x_markings', name2='y_markings',
            result_name=name, xname='x', yname='y',
            category='Markings', label='marking coordinates', xlabel='x', ylabel='y',
            description='coordinates of ant positions, while marking'
        )

        self.exp.write(name)

    def compute_r_phi_marking(self):
        name = 'rphi_markings'
        print('xy_marking_polar')
        self.exp.load(['r', 'phi', 'markings'])

        self.exp.filter_with_time_occurrences(
            name_to_filter='r', filter_name='markings', result_name='r_markings',
            category='Markings', label='marking radial coordinates',
            description='radial coordinates of ant positions, while marking')
        self.exp.write('r_markings')

        self.exp.filter_with_time_occurrences(
            name_to_filter='phi', filter_name='markings', result_name='phi_markings',
            category='Markings', label='marking angular coordinates',
            description='angular coordinates of ant positions, while marking')
        self.exp.write('phi_markings')

        self.exp.add_2d_from_1ds(
            name1='r_markings', name2='phi_markings',
            result_name=name, xname='r', yname='phi',
            category='Markings', label='marking polar coordinates', xlabel='r', ylabel='phi',
            description='polar coordinates of ant positions, while marking'
        )
        self.exp.write(name)

    def compute_marking_interval(self):
        self.exp.load(['markings'])
        name = 'marking_intervals'
        print(name)

        id_exp_ant_array = self.exp.markings.get_index_array_of_id_exp_ant()
        marking_interval_list = []
        for (id_exp, id_ant) in id_exp_ant_array:
            marks = self.__get_marking_of_id_ant_id_exp(id_exp, id_ant)
            marking_interval_list = self.__add_marking_intervals(marking_interval_list, id_ant, id_exp, marks)

        marking_interval_df = self.pd_idx_manager.convert_array_to_df(
                array=marking_interval_list, index_names=[id_exp_name, id_ant_name, id_frame_name], column_names=name)
        self.exp.add_new1d_from_df(
            df=marking_interval_df, name=name, object_type='Events1d', category='Markings',
            label='marking intervals', description='Time intervals between two marking events'
        )
        self.exp.write(name)

    @staticmethod
    def __add_marking_intervals(marking_interval_list, id_ant, id_exp, marks):
        mark_times = marks[:-1, 2]
        lg = len(mark_times)
        id_exp_array = np.full(lg, id_exp)
        id_ant_array = np.full(lg, id_ant)
        marking_interval = marks[1:, 2] - marks[:-1, 2] - 1
        marking_interval_list += list(zip(id_exp_array, id_ant_array, mark_times, marking_interval))
        return marking_interval_list

    def __get_marking_of_id_ant_id_exp(self, id_exp, id_ant):
        marks = np.array(self.exp.markings.get_row(IdxSc[id_exp, id_ant, :]).reset_index())
        return marks

    def compute_marking_distance(self):
        self.exp.load('xy_markings')
        name = 'marking_distances'
        print(name)

        id_exp_ant_array = self.exp.xy_markings.get_index_array_of_id_exp_ant()
        marking_distance_list = []
        for (id_exp, id_ant) in id_exp_ant_array:
            mark_xy = self.__get_marking_xy_of_id_ant_id_exp(id_exp, id_ant)
            marking_distance_list = self.__add_marking_distances(marking_distance_list, id_ant, id_exp, mark_xy)

        marking_distance_df = self.pd_idx_manager.convert_array_to_df(
                array=marking_distance_list, index_names=[id_exp_name, id_ant_name, id_frame_name], column_names=name)

        self.exp.add_new1d_from_df(
            df=marking_distance_df, name=name, object_type='Events1d', category='Markings',
            label='marking distances', description='Distance between two marking events'
        )
        self.exp.write(name)

    def __get_marking_xy_of_id_ant_id_exp(self, id_exp, id_ant):
        res = np.array(self.exp.xy_markings.get_row(IdxSc[id_exp, id_ant, :]).reset_index())
        return res

    @staticmethod
    def __add_marking_distances(marking_distance_list, id_ant, id_exp, mark_xy):
        mark_frames = mark_xy[:-1, 2]
        lg = len(mark_frames)
        id_exp_array = np.full(lg, id_exp)
        id_ant_array = np.full(lg, id_ant)
        marking_distance = distance(mark_xy[1:, -2:], mark_xy[:-1, -2:])
        marking_distance = np.around(marking_distance, 3)
        marking_distance_list += list(zip(id_exp_array, id_ant_array, mark_frames, marking_distance))
        return marking_distance_list
