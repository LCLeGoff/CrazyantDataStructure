import pandas as pd
import numpy as np

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.MiscellaneousTools.Geometry import angle_df, norm_angle_tab, norm_angle_tab2


class AnalyseAntFoodRelation(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'AntFoodRelation'

    def compute_distance2food(self):
        name = 'distance2food'
        print(name)
        food_name_x = 'mm10_food_x'
        food_name_y = 'mm10_food_y'
        name_x = 'mm10_x'
        name_y = 'mm10_y'
        self.exp.load_as_2d(name_x, name_y, result_name='xy', xname='x', yname='y', replace=True)
        self.exp.load_as_2d(food_name_x, food_name_y, result_name='food_xy', xname='x', yname='y', replace=True)

        id_exps = self.exp.get_df(name_x).index.get_level_values(id_exp_name)
        id_ants = self.exp.get_df(name_x).index.get_level_values(id_ant_name)
        frames = self.exp.get_df(name_x).index.get_level_values(id_frame_name)
        idxs = pd.MultiIndex.from_tuples(list(zip(id_exps, frames)), names=[id_exp_name, id_frame_name])

        df_d = self.__reindexing_food_xy(id_ants, idxs)
        df_d = self.__compute_distance_from_food(df_d)

        self.exp.add_new1d_from_df(
            df=df_d, name=name, object_type='TimeSeries1d',
            category=self.category, label='Food to distance', description='Distance between the food and the ants'
        )
        self.exp.write(name)

    def __reindexing_food_xy(self, id_ants, idxs):
        df_d = self.exp.get_df('food_xy').copy()
        df_d = df_d.reindex(idxs)
        df_d[id_ant_name] = id_ants
        df_d.reset_index(inplace=True)
        df_d.columns = [id_exp_name, id_frame_name, 'x', 'y', id_ant_name]
        df_d.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)
        return df_d

    def __compute_distance_from_food(self, df_f):
        df_d = np.around(np.sqrt((df_f.x - self.exp.xy.df.x) ** 2 + (df_f.y - self.exp.xy.df.y) ** 2), 3)
        df_d = pd.DataFrame(df_d)
        return df_d

    def compute_mm10_distance2food(self):
        name = 'distance2food'
        time_window = 10

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=self.category
        )

        self.exp.write(result_name)

    def compute_mm20_distance2food(self):
        name = 'distance2food'
        time_window = 20

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=self.category
        )

        self.exp.write(result_name)

    def compute_is_xy_next2food(self):
        name = 'is_xy_next2food'

        name_distance = 'distance2food'
        self.exp.load(name_distance)
        self.exp.add_copy1d(
            name_to_copy=name_distance, copy_name=name, category=self.category,
            label='Is next to food?', description='Is ants next to the food?'
        )

        neighbor_distance = 15.
        neighbor_distance2 = 5.
        self.exp.operation(name, lambda x: (x < neighbor_distance)*(x > neighbor_distance2))
        self.exp.is_xy_next2food.df = self.exp.is_xy_next2food.df.astype(int)
        self.exp.write(name)

    def compute_xy_next2food(self):
        name = 'xy_next2food'

        self.exp.load('is_xy_next2food')

        self.exp.load_as_2d(name1='mm10_x', name2='mm10_y', xname='x', yname='y', result_name='xy', replace=True)

        self.exp.filter_with_values(
            name_to_filter='xy', filter_name='is_xy_next2food', result_name=name,
            xname='x', yname='y', category=self.category,
            label='XY next to food', xlabel='x', ylabel='y', description='Trajectory of ant next to food'
        )

        self.exp.write(name)

    def compute_speed_xy_next2food(self):
        name = 'speed_xy_next2food'

        self.exp.load(['speed_x', 'speed_y', 'is_xy_next2food'])

        self.exp.add_2d_from_1ds(
            name1='speed_x', name2='speed_y', result_name='dxy'
        )

        self.exp.filter_with_values(
            name_to_filter='dxy', filter_name='is_xy_next2food', result_name=name,
            xname='x', yname='y', category=self.category,
            label='speed vector next to food', xlabel='x', ylabel='y', description='Speed vector of ants next to food'
        )

        self.exp.write(name)

    def compute_speed_next2food(self):
        name = 'speed'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category=self.category, label='speed next to food', description='Instantaneous speed of ant next to food'
        )

        self.exp.write(res_name)

    def compute_mm10_speed_next2food(self):
        name = 'mm10_speed'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category=self.category, label='speed next to food',
            description='Moving mean (time window of 10 frames) of the instantaneous speed of ant close to the food'
        )

        self.exp.write(res_name)

    def compute_mm20_speed_next2food(self):
        name = 'mm20_speed'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category=self.category, label='speed next to food',
            description='Moving mean (time window of 20 frames) of the instantaneous speed of ant close to the food'
        )

        self.exp.write(res_name)

    def compute_distance2food_next2food(self):
        name = 'distance2food'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category=self.category, label='Food distance next to food',
            description='Distance between the food and the ants next to the food'
        )

        self.exp.write(res_name)

    def compute_mm10_distance2food_next2food(self):
        name = 'mm10_distance2food'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category=self.category, label='Food distance next to food',
            description='Moving mean (time window of 10 frames) '
                        'of the distance between the food and the ants next to the food'
        )

        self.exp.write(res_name)

    def compute_mm20_distance2food_next2food(self):
        name = 'mm20_distance2food'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category=self.category, label='Food distance next to food',
            description='Moving mean (time window of 20 frames) '
                        'of the distance between the food and the ants next to the food'
        )

        self.exp.write(res_name)

    def __diff4each_group(self, df: pd.DataFrame):
        name0 = df.columns[0]
        df.dropna(inplace=True)
        id_exp = df.index.get_level_values(id_exp_name)[0]
        d = np.array(df)
        if len(d) > 1:

            d1 = d[1].copy()
            d2 = d[-2].copy()
            d[1:-1] = (d[2:] - d[:-2]) / 2.
            d[0] = d1-d[0]
            d[-1] = d[-1]-d2

            dt = np.array(df.index.get_level_values(id_frame_name), dtype=float)
            dt[1:-1] = dt[2:] - dt[:-2]
            dt[0] = 1
            dt[-1] = 1
            d[dt > 2] = np.nan

            df[name0] = d * self.exp.fps.df.loc[id_exp].fps
        else:
            df[name0] = np.nan

        return df

    def compute_distance2food_next2food_diff(self):
        name = 'distance2food_next2food'
        result_name = 'distance2food_next2food_diff'

        self.exp.load([name, 'fps'])

        self.exp.add_copy(
            old_name=name, new_name=result_name, category=self.category, label='Food distance differential',
            description='Differential of the distance between the food and the ants', replace=True
        )

        self.exp.get_data_object(result_name).change_values(
            self.exp.get_df(result_name).groupby([id_exp_name, id_ant_name]).apply(self.__diff4each_group))

        self.exp.write(result_name)

    def compute_mm10_distance2food_next2food_diff(self):
        name = 'mm10_distance2food_next2food'
        result_name = 'mm10_distance2food_next2food_diff'

        self.exp.load([name, 'fps'])

        self.exp.add_copy(
            old_name=name, new_name=result_name, label='Food distance differential',
            description='Differential of the distance between the food and the ants', replace=True
        )

        self.exp.get_data_object(result_name).change_values(
            self.exp.get_df(result_name).groupby([id_exp_name, id_ant_name]).apply(self.__diff4each_group))

        self.exp.write(result_name)

    def compute_mm20_distance2food_next2food_diff(self):
        name = 'mm20_distance2food_next2food'
        result_name = 'mm20_distance2food_next2food_diff'

        self.exp.load([name, 'fps'])

        self.exp.add_copy(
            old_name=name, new_name=result_name, label='Food distance differential',
            description='Differential of the distance between the food and the ants', replace=True
        )

        self.exp.get_data_object(result_name).change_values(
            self.exp.get_df(result_name).groupby([id_exp_name, id_ant_name]).apply(self.__diff4each_group))

        self.exp.write(result_name)

    def compute_orientation_next2food(self):
        name = 'orientation'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', category=self.category, result_name=res_name,
            label='orientation next to food', description='Body orientation of ant next to food'
        )

        self.exp.write(res_name)

    def compute_mm10_orientation_next2food(self):
        name = 'mm10_orientation'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', category=self.category, result_name=res_name,
            label='orientation next to food',
            description='Moving mean (time window of 10 frames) of the body orientation of ant next to food'
        )

        self.exp.write(res_name)

    def compute_mm20_orientation_next2food(self):
        name = 'mm20_orientation'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            label='orientation next to food', category=self.category,
            description='Moving mean (time window of 20 frames) of the body orientation of ant next to food'
        )

        self.exp.write(res_name)

    def compute_angle_body_food(self):
        name = 'angle_body_food'

        name_x = 'mm10_x'
        name_y = 'mm10_y'
        food_name_x = 'mm10_food_x'
        food_name_y = 'mm10_food_y'

        self.exp.load([food_name_x, food_name_y, 'orientation'])
        self.exp.load_as_2d(name1=name_x, name2=name_y, result_name='xy', xname='x', yname='y', replace=True)
        self.exp.load_as_2d(name1=food_name_x, name2=food_name_y, result_name='food_xy',
                            xname='x', yname='y', replace=True)

        id_exps = self.exp.xy.df.index.get_level_values(id_exp_name)
        id_ants = self.exp.xy.df.index.get_level_values(id_ant_name)
        frames = self.exp.xy.df.index.get_level_values(id_frame_name)
        idxs = pd.MultiIndex.from_tuples(list(zip(id_exps, frames)), names=[id_exp_name, id_frame_name])

        df_food = self.__reindexing_food_xy(id_ants, idxs)

        df_ant_vector = df_food.copy()
        df_ant_vector.x = df_food.x - self.exp.xy.df.x
        df_ant_vector.y = df_food.y - self.exp.xy.df.y
        self.exp.add_copy('orientation', 'ant_food_orientation')
        self.exp.ant_food_orientation.change_values(angle_df(df_ant_vector))

        self.exp.add_copy(
            old_name='orientation', new_name=name, category=self.category, label='Body theta_res to food',
            description='Angle between the ant-food vector and the body vector', replace=True
        )
        self.exp.get_data_object(name).change_values(norm_angle_tab(
            self.exp.ant_food_orientation.df.ant_food_orientation
            - self.exp.orientation.df.orientation))

        self.exp.operation(name, lambda x: np.around(norm_angle_tab2(x), 3))

        self.exp.write(name)

    def compute_mm10_angle_body_food(self):
        name = 'angle_body_food'
        time_window = 10

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=self.category
        )

        self.exp.write(result_name)

    def compute_mm20_angle_body_food(self):
        name = 'angle_body_food'
        time_window = 20

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=self.category
        )

        self.exp.write(result_name)

    def compute_angle_body_food_next2food(self):
        name = 'angle_body_food'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category=self.category, label='orientation next to food',
            description='Angle between the ant-food vector and the body vector for the ants close to the food'
        )

        self.exp.write(res_name)

    def compute_mm10_angle_body_food_next2food(self):
        name = 'mm10_angle_body_food'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category=self.category, label='orientation next to food',
            description='Moving mean (time window of 10 frames)  of the angle'
                        ' between the ant-food vector and the body vector for the ants close to the food'
        )

        self.exp.write(res_name)

    def compute_mm20_angle_body_food_next2food(self):
        name = 'mm20_angle_body_food'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category=self.category, label='orientation next to food',
            description='Moving mean (time window of 20 frames)  of the angle'
                        ' between the ant-food vector and the body vector for the ants close to the food'
        )

        self.exp.write(res_name)
