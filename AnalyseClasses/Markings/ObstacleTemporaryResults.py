import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class ObstacleTemporaryResults:
    def __init__(self, root, group):
        self.pd_idx_manager = PandasIndexManager()
        self.exp = ExperimentGroupBuilder(root).build(group)

    def convert_temporary_result(self, list_id_exp=None):

        list_id_exp = self.exp.set_id_exp_list(list_id_exp)

        name = 'temporary_result'
        temporary_arrival_side_name = 'temporary_arrival_side'
        temporary_departure_side_name = 'temporary_departure_side'
        temporary_same_side_name = 'temporary_same_side'
        self.exp.load([name, 'date'])

        arrival_res = []
        departure_res = []
        same_side = []
        details = []

        same_consecutive_arrival = []
        prev_arrival = None
        same_consecutive_departure = []
        prev_departure = None
        for id_exp in list_id_exp:
            date = self.exp.date.get_value(id_exp)
            if date not in ['10/09/17', '11/09/17', '12/09/17']:
            # if date not in []:
                temp_res = self.exp.temporary_result.get_value(id_exp)
                arrival_res.append((id_exp, self._convert_temp_val(temp_res[0])))
                departure_res.append((id_exp, self._convert_temp_val(temp_res[1])))
                is_side_chosen = int(departure_res[-1][0] != 0)
                is_same_side_chosen = int(arrival_res[-1][1] == departure_res[-1][1])
                same_side.append((id_exp, is_side_chosen*is_same_side_chosen))

                if prev_arrival is not None:
                    if arrival_res[-1][1] in [-1, 1]:
                        same_consecutive_arrival.append((id_exp, int(arrival_res[-1][1] != prev_arrival)))
                prev_arrival = arrival_res[-1][1]

                if prev_departure is not None:
                    if departure_res[-1][1] in [-1, 1]:
                        same_consecutive_departure.append((id_exp, int(departure_res[-1][1] != prev_departure)))
                prev_departure = departure_res[-1][1]

                if temp_res == 'RR':
                    details.append((id_exp, 0))
                elif temp_res == 'LL':
                    details.append((id_exp, 1))
                elif temp_res == 'RL':
                    details.append((id_exp, 2))
                elif temp_res == 'LR':
                    details.append((id_exp, 3))

        arrival_df = pd.DataFrame(arrival_res, columns=['id_exp', temporary_arrival_side_name], dtype=int)
        arrival_df.set_index('id_exp', inplace=True)
        self.exp.add_new1d_from_df(
            df=arrival_df, name=temporary_arrival_side_name, object_type='Characteristics1d',
            category='ObstacleChoice', label='from which side of the obstacle the ant reaches the food',
            description='from which side of the obstacle the ant reaches the food'
        )

        departure_df = pd.DataFrame(departure_res, columns=['id_exp', temporary_departure_side_name], dtype=int)
        departure_df.set_index('id_exp', inplace=True)
        self.exp.add_new1d_from_df(
            df=departure_df, name=temporary_departure_side_name, object_type='Characteristics1d',
            category='ObstacleChoice', label='from which side of the obstacle the ant leaves the food',
            description='from which side of the obstacle the ant leaves the food'
        )

        same_side_df = pd.DataFrame(same_side, columns=['id_exp', temporary_same_side_name], dtype=int)
        same_side_df.set_index('id_exp', inplace=True)
        self.exp.add_new1d_from_df(
            df=same_side_df, name=temporary_same_side_name, object_type='Characteristics1d',
            category='ObstacleChoice', label='Does ant reach and leave the food by the same side?',
            description='Does ant reach and leave the food by the same side?'
        )

        details_df = pd.DataFrame(details, columns=['id_exp', 'details'], dtype=int)
        details_df.set_index('id_exp', inplace=True)
        self.exp.add_new1d_from_df(
            df=details_df, name='details', object_type='Characteristics1d'
        )

        # print('arrival: R vs L')
        # self._binom_test(np.array(arrival_res)[:, 1], [1, -1])
        #
        # print('departure: R vs L')
        # self._binom_test(np.array(departure_res)[:, 1], [1, -1])
        #
        # print('same choice vs not same')
        # self._binom_test(np.array(same_side)[:, 1], [1, 0])
        #
        # print('consecutive arrival')
        # self._binom_test(np.array(same_consecutive_arrival)[:, 1], [1, 0])
        #
        # print('consecutive departure')
        # self._binom_test(np.array(same_consecutive_departure)[:, 1], [1, 0])

        self.exp.write([temporary_arrival_side_name, temporary_departure_side_name, temporary_same_side_name])

        fig, ax = self.exp.details.plotter.hist1d(
            bins=np.arange(-0.5, 4, 1.), ls='', marker='o', confidence=True)
        # ax.set_ylim((0, 30))
        ax.set_xlim((-0.5, 1.5))
        ax.set_xticks([-1, 0, 1, 2, 3, 4])
        ax.set_xticklabels(['', 'RR', 'LL', 'RL', 'LR'])
        # ax.set_yticks(range(0, 32, 2))
        ax.set_yticks(range(0, 65, 5))
        ax.grid()
        # plt.show()

    @staticmethod
    def _binom_test(choices, vals):
        # choices = np.array(arrival_res)[:, 1]
        n_r = np.sum(choices == vals[0])
        n_l = np.sum(choices == vals[1])
        print(n_r, n_l, scs.binom_test([n_r, n_l]))

    @staticmethod
    def _convert_temp_val(temp_res):
        if temp_res == 'R':
            return 1
        elif temp_res == 'L':
            return -1
        else:
            return 0
