import numpy as np
import pandas as pd

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Tools.MiscellaneousTools.Geometry import angle_df, norm_angle_tab, norm_angle_tab2


class AnalyseBaseFood:
    def __init__(self, root, group):
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

    def compute_speed_food(self):
        name = 'food_speed'
        self.exp.load(['food_x', 'food_y', 'fps'])

        self.exp.add_copy1d(
            name_to_copy='food_x', copy_name=name, category='Trajectory', label='Food speed',
            description='Instantaneous speed of the food'
        )
        # self.exp.add_copy1d(
        #     name_to_copy='food_x', copy_name=name+'_x', category='Trajectory', label='X food speed',
        #     description='X coordinate of the instantaneous speed of the ants'
        # )
        # self.exp.add_copy1d(
        #     name_to_copy='food_x', copy_name=name+'_y', category='Trajectory', label='Y food speed',
        #     description='Y coordinate of the instantaneous speed of the food'
        # )

        for id_exp in self.exp.characteristic_timeseries_exp_frame_index:

            dx = np.array(self.exp.food_x.df.loc[id_exp, :])
            dx[1:-1, :] = (dx[2:, :]-dx[:-2, :])/2.
            dx[0, :] = dx[1, :]-dx[0, :]
            dx[-1, :] = dx[-1, :]-dx[-2, :]

            dy = np.array(self.exp.food_y.df.loc[id_exp, :])
            dy[1:-1, :] = (dy[2:, :]-dy[:-2, :])/2.
            dy[0, :] = dy[1, :]-dy[0, :]
            dy[-1, :] = dy[-1, :]-dy[-2, :]

            dt = np.array(self.exp.characteristic_timeseries_exp_frame_index[id_exp], dtype=float)
            dt.sort()
            dt[0] = 1
            dt[-1] = 1
            dt[1:-1] = dt[2:]-dt[:-2]
            dx[dt > 2] = np.nan
            dy[dt > 2] = np.nan

            # self.exp.food_speed_x.df.loc[id_exp, id_ant, :] = dx*self.exp.fps.df.loc[id_exp].fps
            # self.exp.food_speed_y.df.loc[id_exp, id_ant, :] = dy*self.exp.fps.df.loc[id_exp].fps
            self.exp.food_speed.df.loc[id_exp, :] = np.sqrt(dx**2+dy**2)*self.exp.fps.df.loc[id_exp].fps

        self.exp.write([name])

    def compute_is_xy_next_to_food(self):
        name = 'is_xy_next_to_food'

        name_distance = 'distance_to_food'
        self.exp.load(name_distance)
        self.exp.add_copy1d(
            name_to_copy=name_distance, copy_name=name, category='FoodBase',
            label='Is next to food?', description='Is ants next to the food?'
        )

        neighbor_distance = 15.
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

    def compute_speed_xy_next_to_food(self):
        name = 'speed_xy_next_to_food'

        self.exp.load(['speed_x', 'speed_y', 'is_xy_next_to_food'])

        self.exp.add_2d_from_1ds(
            name1='speed_x', name2='speed_y', result_name='dxy'
        )

        self.exp.filter_with_values(
            name_to_filter='dxy', filter_name='is_xy_next_to_food', result_name=name,
            xname='x', yname='y', category='BaseFood',
            label='speed vector next to food', xlabel='x', ylabel='y', description='Speed vector of ants next to food'
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

    def compute_orientation_to_food(self):
        name = 'orientation_to_food'

        self.exp.load(['food_x', 'food_y', 'xy_next_to_food', 'orientation_next_to_food'])

        id_exps = self.exp.xy_next_to_food.df.index.get_level_values('id_exp')
        id_ants = self.exp.xy_next_to_food.df.index.get_level_values('id_ant')
        frames = self.exp.xy_next_to_food.df.index.get_level_values('frame')
        idxs = pd.MultiIndex.from_tuples(list(zip(id_exps, frames)), names=['id_exp', 'frames'])
        self.exp.add_2d_from_1ds(
            name1='food_x', name2='food_y', result_name='food_xy',
            xname='x_ant', yname='y_ant'
        )
        df_food = self.__reindexing_food_xy(id_ants, idxs)

        df_ant_vector = df_food.copy()
        df_ant_vector.x = df_food.x - self.exp.xy_next_to_food.df.x
        df_ant_vector.y = df_food.y - self.exp.xy_next_to_food.df.y
        self.exp.add_copy('orientation_next_to_food', 'ant_food_orientation')
        self.exp.ant_food_orientation.change_values(angle_df(df_ant_vector))

        self.exp.add_copy('orientation_next_to_food', name)
        self.exp.rename(
            old_name=name, new_name=name, category='BaseFood', label='Body theta_res to food',
            description='Angle between the ant-food vector and the body theta_res vector'
        )
        self.exp.orientation_to_food.change_values(norm_angle_tab(
            self.exp.ant_food_orientation.df.ant_food_orientation
            - self.exp.orientation_next_to_food.df.orientation_next_to_food))
        self.exp.operation('orientation_to_food', lambda x: norm_angle_tab2(x))

        self.exp.write(name)
