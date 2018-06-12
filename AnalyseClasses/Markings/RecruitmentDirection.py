import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Tools.MiscellaneousTools.Fits import linear_fit
from Tools.MiscellaneousTools.Geometry import convert_polar2cartesian, convert_cartesian2polar
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class RecruitmentDirection:
    def __init__(self, root, group):
        self.pd_idx_manager = PandasIndexManager()
        self.exp = ExperimentGroupBuilder(root).build(group)
        self.circular_arena_radius = 120

    def compute_recruitment_direction(self, list_id_exp=None):

        recruitment_intervals_name = 'recruitment_intervals'
        xy_markings_name = 'xy_markings'
        r_markings_name = 'r_markings'
        phi_markings_name = 'phi_markings'

        recruitment_interval_filter_name = 'recruitment_interval_filter'
        is_in_circular_arena_name = 'is_in_circular_arena'

        xy_recruitments_name = 'xy_recruitments'
        r_recruitments_name = 'r_recruitments'
        phi_recruitments_name = 'phi_recruitments'

        self.exp.load([recruitment_intervals_name, r_markings_name, phi_markings_name, xy_markings_name])

        self._compute_recruitment_interval_filter(
            r_markings_name, recruitment_intervals_name, recruitment_interval_filter_name)

        self._compute_is_in_circular_arena_filter(
            is_in_circular_arena_name, r_markings_name, recruitment_interval_filter_name)

        self._extract_recruitment_period_in_circular_arena_for(xy_markings_name, xy_recruitments_name)
        self._extract_recruitment_period_in_circular_arena_for(r_markings_name, r_recruitments_name)
        self._extract_recruitment_period_in_circular_arena_for(phi_markings_name, phi_recruitments_name)

        self._compute_ab_recruitment(recruitment_interval_filter_name, list_id_exp)

    def _compute_ab_recruitment(self, recruitment_interval_filter_name, list_id_exp):
        if list_id_exp is None:
            list_id_exp = self.exp.r_recruitments.get_index_array_of_id_exp()
        list_id_exp_ant = self.exp.r_recruitments.get_index_array_of_id_exp_ant()

        res = []
        for id_exp, id_ant in list_id_exp_ant:
            if id_exp in list_id_exp:

                r_array, phi_array, recruitment_interval_index_array =\
                    self._get_r_phi_and_recruitment_interval_index_arrays(
                        id_exp, id_ant, recruitment_interval_filter_name)

                filter_value_set = set(recruitment_interval_index_array[:, -1])
                for filter_val in filter_value_set:
                    a, b = self._compute_fit_coefficient_of_recruitment_markings(
                        filter_val, r_array, phi_array, recruitment_interval_index_array)

                    res.append((id_exp, id_ant, filter_val, a, b))

        self._add_to_exp_ab_recruitment(res)

    def _add_to_exp_ab_recruitment(self, res):
        idx_names = ['id_exp', 'id_ant', 'frame']
        df = pd.DataFrame(res, columns=idx_names + ['a', 'b'])
        df.set_index(idx_names, inplace=True)
        self.exp.add_new2d_from_df(
            df=df, name='ab_recruitment', xname='a', yname='b', object_type='Events2d',
            category='Recruitment', label='coefficient fit of recruitment',
            xlabel='a', ylabel='b', description='Coefficient of the linear regression of the recruitment markings')

    def _compute_fit_coefficient_of_recruitment_markings(self, filter_val, r_array, phi_array,
                                                         recruitment_interval_index_array):
        r, phi = self._get_rphi_from_filter_val(
            filter_val, r_array, phi_array, recruitment_interval_index_array)
        # ax = plt.subplot(111, projection='polar')
        # ax.plot(phi, r, 'o-')
        phi, phi0 = self._rotation_of_markings(phi)
        # ax.plot(phi, r, 'o-')
        x_fit, y_fit = self._linear_fit_of_the_rotated_markings(r, phi)
        phi_fit, r_fit = self._rotation_of_the_fit_line(phi0, x_fit, y_fit)
        # ax.plot(phi_fit, r_fit)
        a, b = self._compute_ab_of_the_fit_line(r_fit, phi_fit)
        # plt.show()
        return a, b

    def _get_r_phi_and_recruitment_interval_index_arrays(self, id_exp, id_ant, recruitment_interval_filter_name):
        phi_df = self.exp.phi_recruitments.get_row_of_id_exp_ant(id_exp, id_ant)
        phi_array = self.exp.pandas_index_manager.convert_df_to_array(phi_df)
        r_df = self.exp.r_recruitments.get_row_of_id_exp_ant(id_exp, id_ant)
        r_array = self.exp.pandas_index_manager.convert_df_to_array(r_df)
        recruitment_interval_index_df = \
            self.exp.__dict__[recruitment_interval_filter_name].get_row_of_id_exp_ant(id_exp, id_ant)
        recruitment_interval_index_array = \
            self.exp.pandas_index_manager.convert_df_to_array(recruitment_interval_index_df)
        return r_array, phi_array, recruitment_interval_index_array

    @staticmethod
    def _rotation_of_the_fit_line(phi0, x_fit, y_fit):
        r_fit, phi_fit = convert_cartesian2polar(x_fit, y_fit)
        phi_fit += phi0
        return phi_fit, r_fit

    @staticmethod
    def _linear_fit_of_the_rotated_markings(r, phi):
        x, y = convert_polar2cartesian(r, phi)
        a, b, x_fit, y_fit = linear_fit(x, y)
        return x_fit, y_fit

    @staticmethod
    def _compute_ab_of_the_fit_line(r, phi):
        x, y = convert_polar2cartesian(r, phi)
        a = (y[1]-y[0])/float(x[1]-x[0])
        b = y[0]-a*x[0]
        return a, b

    @staticmethod
    def _rotation_of_markings(phi):
        phi0 = phi[-1]
        phi -= phi0
        return phi, phi0

    def _compute_recruitment_interval_filter(self, r_markings, recruitment_intervals, recruitment_interval_index_name):
        res_name, interval_index_name = self.exp.filter_with_time_intervals(
            name_to_filter=r_markings,
            name_intervals=recruitment_intervals
        )
        self.exp.rename_object(interval_index_name, recruitment_interval_index_name)

    def _extract_recruitment_period_in_circular_arena_for(self, markings_name, recruitments_name):
        self.exp.filter_with_values(
            name_to_filter=markings_name, filter_name='is_in_circular_arena',
            result_name=markings_name + '_in_circular_arena'
        )
        result_name, interval_index_name = self.exp.filter_with_time_intervals(
            name_to_filter=markings_name + '_in_circular_arena',
            name_intervals='recruitment_intervals',
            result_name=recruitments_name)

        return result_name, interval_index_name

    def _compute_is_in_circular_arena_filter(
            self, is_in_circular_arena, r_markings, recruitment_interval_filter_name):

        self._compute_when_markings_inside_circular_arena(r_markings, is_in_circular_arena)

        self.exp.filter_with_time_intervals(
            name_to_filter=is_in_circular_arena, name_intervals='recruitment_intervals',
            result_name='temp', replace=True
        )
        self.exp.is_in_circular_arena.df = self.exp.temp.df
        self.exp.remove_object('temp')

        is_in_circular_arena_df = self.exp.is_in_circular_arena.df
        interval_index_df = self.exp.__dict__[recruitment_interval_filter_name].df

        array_id_exp_ant_frame = self.exp.is_in_circular_arena.get_index_array_of_id_exp_ant_frame()
        (id_exp, id_ant, frame) = array_id_exp_ant_frame[0, :]
        prev_is_it_inside = is_in_circular_arena_df.loc[(id_exp, id_ant, frame), is_in_circular_arena_df.columns[0]]
        prev_interval = int(np.array(interval_index_df.loc[(id_exp, id_ant, frame), interval_index_df.columns[0]]))
        for id_exp, id_ant, frame in array_id_exp_ant_frame:
            current_interval = int(np.array(interval_index_df.loc[
                                                (id_exp, id_ant, frame), interval_index_df.columns[0]]))

            is_same_interval_than_prev = prev_interval == current_interval
            if is_same_interval_than_prev:
                is_in_circular_arena_df.loc[(id_exp, id_ant, frame), is_in_circular_arena_df.columns[0]]\
                    *= prev_is_it_inside

            prev_is_it_inside = is_in_circular_arena_df.loc[(id_exp, id_ant, frame), is_in_circular_arena_df.columns[0]]
            prev_interval = current_interval

        self.exp.is_in_circular_arena.df = is_in_circular_arena_df

        self.exp.filter_with_values(
            name_to_filter=recruitment_interval_filter_name, filter_name=is_in_circular_arena,
            result_name=recruitment_interval_filter_name, redo=True
        )

        # if is_in_circular_arena_df.loc[(id_exp, id_ant, frame)] == 1:
        #     if already_exit == 1:
        #         is_same_interval_than_prev = prev_interval == df_recruitment_interval.loc[(id_exp, id_ant, frame)]
        #         if is_same_interval_than_prev:
        #             is_in_circular_arena_df.loc[(id_exp, id_ant, frame)] = 0
        #         else:
        #             already_exit = 0
        # else:
        #     already_exit = 1
        # prev_interval = df_recruitment_interval.loc[(id_exp, id_ant, frame)]

    def _compute_when_markings_inside_circular_arena(self, r_markings, is_in_circular_arena):
        self.exp.add_copy1d(name_to_copy=r_markings, copy_name=is_in_circular_arena)
        self.exp.operation(
            is_in_circular_arena,
            lambda x: (x < self.circular_arena_radius).astype(int))

    @staticmethod
    def _get_rphi_from_filter_val(filter_val, r_array, phi_array, interval_index_array):
        idx_where_filter_val = np.where(interval_index_array[:, -1] == filter_val)[0]

        temp_r_array = r_array[idx_where_filter_val, :]
        r = temp_r_array[:, -1]

        temp_phi_array = phi_array[idx_where_filter_val, :]
        phi = temp_phi_array[:, -1]
        return r, phi

    @staticmethod
    def _get_val_from_filter_val(filter_val, xy_array):
        idx_where_filter_val = np.where(xy_array[:, -3] == filter_val)[0]
        temp_xy_array = xy_array[idx_where_filter_val, :]
        x = temp_xy_array[:, -2]
        y = temp_xy_array[:, -1]
        return x, y

        #
        # self.exp.fit(
        #     name_to_fit=xy_recruitments_name, filter_name=interval_index_name, level='ant', filter_as_frame=True
        # )

        # delta_phi_recruitment = 'delta_'+phi_recruitments
        # mean_delta_phi_recruitments = 'mean_'+delta_phi_recruitment
        # mean_start_recruitments = 'mean_start_recruitments'

        # self.exp.compute_delta(
        #     name_to_delta=phi_recruitments,
        #     filter_name=interval_index_name,
        #     result_name=delta_phi_recruitment
        # )
        #
        # self.exp.compute_mean_in_time_interval(
        #     name_to_average=delta_phi_recruitment,
        #     name_intervals=recruitment_intervals,
        #     result_name=mean_delta_phi_recruitments)
        #
        # self.exp.filter_with_time_occurrences(
        #     name_to_filter=phi,
        #     filter_name=mean_delta_phi_recruitments,
        #     result_name=mean_start_recruitments)
        #
        # self.exp.operation_between_2names(
        #     mean_delta_phi_recruitments, mean_start_recruitments, lambda x, y: x+y)
        # # self.exp.write(result_name)
        #
        # result_name = self.exp.compute_mean_in_time_interval(
        #     name_to_average=phi_markings, name_intervals=recruitment_intervals)
        # print(result_name)
        #
        # id_exp_list = self.exp.__dict__[result_name].get_index_array_of_id_exp()
        #
        # id_exp_ant_frame_array = self.exp.__dict__[result_name].get_index_array_of_id_exp_ant_frame()
        #
        # index_list = []
        # for id_exp in id_exp_list:
        #     temp_index_array = id_exp_ant_frame_array[id_exp_ant_frame_array[:, 0] == id_exp, :]
        #     frame_array = temp_index_array[:, 2]
        #     idx_frame_min = frame_array.argmin()
        #     idx_min = tuple(temp_index_array[idx_frame_min, :])
        #     index_list.append(idx_min)
        #
        # self.exp.add_copy1d(
        #     name_to_copy=result_name, copy_name='first_recruitment'
        # )
        # self.exp.first_recruitment.df = self.exp.__dict__[result_name].get_row_of_idx_array(index_list)
