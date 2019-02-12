import numpy as np
import pandas as pd

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class AnalyseBaseFood:
    def __init__(self, root, group):
        self.pd_idx_manager = PandasIndexManager()
        self.exp = ExperimentGroupBuilder(root).build(group)

    def compute_distance_to_food(self):
        name = 'distance_to_food'
        print(name)
        self.exp.load(['food_x', 'food_y', 'x', 'y'])

        id_exps = self.exp.x.df.index.get_level_values('id_exp')
        id_ants = self.exp.x.df.index.get_level_values('id_ant')
        frames = self.exp.x.df.index.get_level_values('frame')
        idxs = pd.MultiIndex.from_tuples(list(zip(id_exps, frames)), names=['id_exp', 'frames'])

        self.exp.add_2d_from_1ds(
            name1='food_x', name2='food_y', result_name='food_xy',
            xname='x', yname='y'
        )

        df_d = self.__reindexing_food_xy(id_ants, idxs)
        df_d = self.__compute_distance_from_food(df_d)

        self.exp.add_new1d_from_df(
            df=df_d, name=name, object_type='TimeSeries1d',
            category='FoodBase', label='Food to distance', description='Distance between the food and the ants'
        )
        self.exp.write(name)

    def __reindexing_food_xy(self, id_ants, idxs):
        df_d = self.exp.food_xy.df.copy()
        df_d = df_d.loc[idxs]
        df_d['id_ant'] = id_ants
        df_d.reset_index(inplace=True)
        df_d.columns = ['id_exp', 'frame', 'x', 'y', 'id_ant']
        df_d.set_index(['id_exp', 'id_ant', 'frame'], inplace=True)
        return df_d

    def __compute_distance_from_food(self, df_f):
        df_d = np.sqrt((df_f.x - self.exp.x.df.x) ** 2 + (df_f.y - self.exp.y.df.y) ** 2)
        df_d = pd.DataFrame(df_d)
        return df_d

    def compute_is_xy_next_to_food(self):
        name = 'is_xy_next_to_food'

        name_distance = 'distance_to_food'
        self.exp.load(name_distance)
        self.exp.add_copy1d(
            name_to_copy=name_distance, copy_name=name, category='FoodBase',
            label='Is next to food?', description='Is ants next to the food?'
        )

        neighbor_distance = 30.
        self.exp.operation(name, lambda x: x < neighbor_distance)
        self.exp.is_xy_next_to_food.df = self.exp.is_xy_next_to_food.df.astype(int)
        self.exp.write(name)

    def compute_xy_next_to_food(self):
        name = 'xy_next_to_food'

        self.exp.load(['x', 'y', 'is_xy_next_to_food'])

        self.exp.add_2d_from_1ds(
            name1='x', name2='y', result_name='xy'
        )

        self.exp.filter_with_values(
            name_to_filter='xy', filter_name='is_xy_next_to_food', result_name=name,
            xname='x', yname='y', category='BaseFood',
            label='XY next to food', xlabel='x', ylabel='y', description='Trajectory of ant next to food'
        )

        self.exp.write(name)

    def compute_dxy_next_to_food(self):
        name = 'dxy_next_to_food'

        self.exp.load(['dx', 'dy', 'is_xy_next_to_food'])

        self.exp.add_2d_from_1ds(
            name1='dx', name2='dy', result_name='dxy'
        )

        self.exp.filter_with_values(
            name_to_filter='xy', filter_name='is_xy_next_to_food', result_name=name,
            xname='x', yname='y', category='BaseFood',
            label='dXY next to food', xlabel='x', ylabel='y', description='Trajectory of ant next to food'
        )

        self.exp.write(name)

    def compute_speed_next_to_food(self):
        name = 'speed'
        res_name = name+'_next_to_food'

        self.exp.load([name, 'is_xy_next_to_food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next_to_food', result_name=res_name,
            category='BaseFood', label='speed next to food', description='Instantaneous speed of ant next to food'
        )

        self.exp.write(res_name)

    def compute_orientation_next_to_food(self):
        name = 'orientation'
        res_name = name + '_next_to_food'

        self.exp.load([name, 'is_xy_next_to_food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next_to_food', result_name=res_name,
            category='BaseFood', label='orientation next to food', description='Body orientation of ant next to food'
        )

        self.exp.write(res_name)
