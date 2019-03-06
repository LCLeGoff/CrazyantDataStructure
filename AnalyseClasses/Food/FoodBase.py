import numpy as np
import pandas as pd
from sklearn import svm

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from ExperimentGroups import ExperimentGroups
from Tools.MiscellaneousTools.Geometry import angle_df, norm_angle_tab, norm_angle_tab2


class AnalyseFoodBase:
    def __init__(self, root, group, exp: ExperimentGroups = None):
        if exp is None:
            self.exp = ExperimentGroupBuilder(root).build(group)
        else:
            self.exp = exp

    def compute_distance2food(self):
        name = 'distance2food'
        print(name)
        self.exp.load(['food_x', 'food_y', 'x', 'y'])

        id_exps = self.exp.x.df.index.get_level_values('id_exp')
        id_ants = self.exp.x.df.index.get_level_values('id_ant')
        frames = self.exp.x.df.index.get_level_values('frame')
        idxs = pd.MultiIndex.from_tuples(list(zip(id_exps, frames)), names=['id_exp', 'frame'])

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
        df_d = df_d.reindex(idxs)
        df_d['id_ant'] = id_ants
        df_d.reset_index(inplace=True)
        df_d.columns = ['id_exp', 'frame', 'x', 'y', 'id_ant']
        df_d.set_index(['id_exp', 'id_ant', 'frame'], inplace=True)
        return df_d

    def __compute_distance_from_food(self, df_f):
        df_d = np.around(np.sqrt((df_f.x - self.exp.x.df.x) ** 2 + (df_f.y - self.exp.y.df.y) ** 2), 3)
        df_d = pd.DataFrame(df_d)
        return df_d

    def compute_mm5_distance2food(self):
        name = 'distance2food'
        category = 'Distance2foodMM'
        time_window = 5

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=category
        )

        self.exp.write(result_name)

    def compute_mm10_distance2food(self):
        name = 'distance2food'
        category = 'Distance2foodMM'
        time_window = 10

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=category
        )

        self.exp.write(result_name)

    def compute_mm20_distance2food(self):
        name = 'distance2food'
        category = 'Distance2foodMM'
        time_window = 20

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=category
        )

        self.exp.write(result_name)

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
            dx1 = dx[1, :]
            dx2 = dx[-2, :]
            dx[1:-1, :] = (dx[2:, :]-dx[:-2, :])/2.
            dx[0, :] = dx1 - dx[0, :]
            dx[-1, :] = dx[-1, :] - dx2

            dy = np.array(self.exp.food_y.df.loc[id_exp, :])
            dy1 = dy[1, :]
            dy2 = dy[-2, :]
            dy[1:-1, :] = (dy[2:, :]-dy[:-2, :])/2.
            dy[0, :] = dy1 - dy[0, :]
            dy[-1, :] = dy[-1, :] - dy2

            dt = np.array(self.exp.characteristic_timeseries_exp_frame_index[id_exp], dtype=float)
            dt.sort()
            dt[1:-1] = dt[2:]-dt[:-2]
            dt[0] = 1
            dt[-1] = 1
            dx[dt > 2] = np.nan
            dy[dt > 2] = np.nan

            # self.exp.food_speed_x.df.loc[id_exp, id_ant, :] = dx*self.exp.fps.df.loc[id_exp].fps
            # self.exp.food_speed_y.df.loc[id_exp, id_ant, :] = dy*self.exp.fps.df.loc[id_exp].fps
            self.exp.food_speed.df.loc[id_exp, :] = np.around(np.sqrt(dx**2+dy**2)*self.exp.fps.df.loc[id_exp].fps, 3)

        self.exp.write([name])

    def compute_is_xy_next2food(self):
        name = 'is_xy_next2food'

        name_distance = 'distance2food'
        self.exp.load(name_distance)
        self.exp.add_copy1d(
            name_to_copy=name_distance, copy_name=name, category='FoodBase',
            label='Is next to food?', description='Is ants next to the food?'
        )

        neighbor_distance = 15.
        neighbor_distance2 = 5.
        self.exp.operation(name, lambda x: (x < neighbor_distance)*(x > neighbor_distance2))
        self.exp.is_xy_next2food.df = self.exp.is_xy_next2food.df.astype(int)
        self.exp.write(name)

    def compute_xy_next2food(self):
        name = 'xy_next2food'

        self.exp.load(['x', 'y', 'is_xy_next2food'])

        self.exp.add_2d_from_1ds(
            name1='x', name2='y', result_name='xy'
        )

        self.exp.filter_with_values(
            name_to_filter='xy', filter_name='is_xy_next2food', result_name=name,
            xname='x', yname='y', category='FoodBase',
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
            xname='x', yname='y', category='FoodBase',
            label='speed vector next to food', xlabel='x', ylabel='y', description='Speed vector of ants next to food'
        )

        self.exp.write(name)

    def compute_speed_next2food(self):
        name = 'speed'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='speed next to food', description='Instantaneous speed of ant next to food'
        )

        self.exp.write(res_name)

    def compute_mm10_speed_next2food(self):
        name = 'mm10_speed'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='speed next to food',
            description='Moving mean (time window of 10 frames) of the instantaneous speed of ant close to the food'
        )

        self.exp.write(res_name)

    def compute_mm20_speed_next2food(self):
        name = 'mm20_speed'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='speed next to food',
            description='Moving mean (time window of 20 frames) of the instantaneous speed of ant close to the food'
        )

        self.exp.write(res_name)

    def compute_distance2food_next2food(self):
        name = 'distance2food'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='Food distance next to food',
            description='Distance between the food and the ants next to the food'
        )

        self.exp.write(res_name)

    def compute_mm5_distance2food_next2food(self):
        name = 'mm5_distance2food'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='Food distance next to food',
            description='Moving mean (time window of 5 frames) '
                        'of the distance between the food and the ants next to the food'
        )

        self.exp.write(res_name)

    def compute_mm10_distance2food_next2food(self):
        name = 'mm10_distance2food'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='Food distance next to food',
            description='Moving mean (time window of 10 frames) '
                        'of the distance between the food and the ants next to the food'
        )

        self.exp.write(res_name)

    def compute_mm20_distance2food_next2food(self):
        name = 'mm20_distance2food'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='Food distance next to food',
            description='Moving mean (time window of 20 frames) '
                        'of the distance between the food and the ants next to the food'
        )

        self.exp.write(res_name)

    def __diff4each_group(self, df: pd.DataFrame):
        name0 = df.columns[0]
        df.dropna(inplace=True)
        id_exp = df.index.get_level_values('id_exp')[0]
        d = np.array(df)
        if len(d) > 1:

            d1 = d[1].copy()
            d2 = d[-2].copy()
            d[1:-1] = (d[2:] - d[:-2]) / 2.
            d[0] = d1-d[0]
            d[-1] = d[-1]-d2

            dt = np.array(df.index.get_level_values('frame'), dtype=float)
            dt[1:-1] = dt[2:] - dt[:-2]
            dt[0] = 1
            dt[-1] = 1
            d[dt > 2] = np.nan

            df[name0] = d * self.exp.fps.df.loc[id_exp].fps
        else:
            df[name0] = np.nan

        return df

    def compute_distance2food_next2food_differential(self):
        name = 'distance2food_next2food'
        result_name = 'distance2food_next2food_diff'

        self.exp.load([name, 'fps'])

        self.exp.add_copy(
            old_name=name, new_name=result_name, category='FoodBase', label='Food distance differential',
            description='Differential of the distance between the food and the ants', replace=True
        )

        self.exp.get_data_object(result_name).change_values(
            self.exp.get_df(result_name).groupby(['id_exp', 'id_ant']).apply(self.__diff4each_group))

        self.exp.write(result_name)

    def compute_mm10_distance2food_next2food_differential(self):
        name = 'mm10_distance2food_next2food'
        result_name = 'mm10_distance2food_next2food_diff'

        self.exp.load([name, 'fps'])

        self.exp.add_copy(
            old_name=name, new_name=result_name, category='FoodBase', label='Food distance differential',
            description='Differential of the distance between the food and the ants', replace=True
        )

        self.exp.get_data_object(result_name).change_values(
            self.exp.get_df(result_name).groupby(['id_exp', 'id_ant']).apply(self.__diff4each_group))

        self.exp.write(result_name)

    def compute_mm20_distance2food_next2food_differential(self):
        name = 'mm20_distance2food_next2food'
        result_name = 'mm20_distance2food_next2food_diff'

        self.exp.load([name, 'fps'])

        self.exp.add_copy(
            old_name=name, new_name=result_name, category='FoodBase', label='Food distance differential',
            description='Differential of the distance between the food and the ants', replace=True
        )

        self.exp.get_data_object(result_name).change_values(
            self.exp.get_df(result_name).groupby(['id_exp', 'id_ant']).apply(self.__diff4each_group))

        self.exp.write(result_name)

    def compute_orientation_next2food(self):
        name = 'orientation'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='orientation next to food', description='Body orientation of ant next to food'
        )

        self.exp.write(res_name)

    def compute_mm10_orientation_next2food(self):
        name = 'mm10_orientation'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='orientation next to food',
            description='Moving mean (time window of 10 frames) of the body orientation of ant next to food'
        )

        self.exp.write(res_name)

    def compute_mm20_orientation_next2food(self):
        name = 'mm20_orientation'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='orientation next to food',
            description='Moving mean (time window of 20 frames) of the body orientation of ant next to food'
        )

        self.exp.write(res_name)

    def compute_angle_body_food(self):
        name = 'angle_body_food'

        self.exp.load(['food_x', 'food_y', 'x', 'y', 'orientation'])

        self.exp.add_2d_from_1ds(
            name1='x', name2='y', result_name='xy',
            xname='x', yname='y', replace=True
        )
        id_exps = self.exp.xy.df.index.get_level_values('id_exp')
        id_ants = self.exp.xy.df.index.get_level_values('id_ant')
        frames = self.exp.xy.df.index.get_level_values('frame')
        idxs = pd.MultiIndex.from_tuples(list(zip(id_exps, frames)), names=['id_exp', 'frame'])
        self.exp.add_2d_from_1ds(
            name1='food_x', name2='food_y', result_name='food_xy',
            xname='x_ant', yname='y_ant', replace=True
        )
        df_food = self.__reindexing_food_xy(id_ants, idxs)

        df_ant_vector = df_food.copy()
        df_ant_vector.x = df_food.x - self.exp.xy.df.x
        df_ant_vector.y = df_food.y - self.exp.xy.df.y
        self.exp.add_copy('orientation', 'ant_food_orientation')
        self.exp.ant_food_orientation.change_values(angle_df(df_ant_vector))

        self.exp.add_copy(
            old_name='orientation', new_name=name, category='FoodBase', label='Body theta_res to food',
            description='Angle between the ant-food vector and the body vector', replace=True
        )
        self.exp.get_data_object(name).change_values(norm_angle_tab(
            self.exp.ant_food_orientation.df.ant_food_orientation
            - self.exp.orientation.df.orientation))
        self.exp.operation(name, lambda x: np.around(norm_angle_tab2(x), 3))

        self.exp.write(name)

    def compute_mm5_angle_body_food(self):
        name = 'angle_body_food'
        category = 'Distance2foodMM'
        time_window = 5

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=category
        )

        self.exp.write(result_name)

    def compute_mm10_angle_body_food(self):
        name = 'angle_body_food'
        category = 'Distance2foodMM'
        time_window = 10

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=category
        )

        self.exp.write(result_name)

    def compute_mm20_angle_body_food(self):
        name = 'angle_body_food'
        category = 'Distance2foodMM'
        time_window = 20

        self.exp.load(name)
        result_name = self.exp.moving_mean4exp_ant_frame_indexed_1d(
            name_to_average=name, time_window=time_window, category=category
        )

        self.exp.write(result_name)

    def compute_angle_body_food_next2food(self):
        name = 'angle_body_food'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='orientation next to food',
            description='Angle between the ant-food vector and the body vector for the ants close to the food'
        )

        self.exp.write(res_name)

    def compute_mm5_angle_body_food_next2food(self):
        name = 'mm5_angle_body_food'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='orientation next to food',
            description='Moving mean (time window of 5 frames)  of the angle'
                        ' between the ant-food vector and the body vector for the ants close to the food'
        )

        self.exp.write(res_name)

    def compute_mm10_angle_body_food_next2food(self):
        name = 'mm10_angle_body_food'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category='FoodBase', label='orientation next to food',
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
            category='FoodBase', label='orientation next to food',
            description='Moving mean (time window of 20 frames)  of the angle'
                        ' between the ant-food vector and the body vector for the ants close to the food'
        )

        self.exp.write(res_name)

    def compute_is_carrying(self):
        name_result = 'carrying_next2food'

        speed_name = 'mm20_speed_next2food'
        orientation_name = 'mm20_angle_body_food_next2food'
        distance_name = 'mm20_distance2food_next2food'
        distance_diff_name = 'mm20_distance2food_next2food_diff'
        training_set_name = 'carrying_training_set'

        self.exp.load(
            [training_set_name, speed_name, orientation_name, distance_name, distance_diff_name, 'mm2px'])

        self.exp.get_data_object(orientation_name).change_values(self.exp.get_df(orientation_name).abs())

        df_features, df_labels = self.__get_training_features_and_labels4carrying(
            speed_name, orientation_name, distance_name, distance_diff_name, training_set_name)

        df_to_predict = self.__get_df_to_predict_carrying(
            distance_diff_name, distance_name, orientation_name, speed_name)

        clf = svm.SVC(kernel='rbf', gamma='auto')
        clf.fit(df_features.loc[pd.IndexSlice[1, :, :], :], df_labels.loc[pd.IndexSlice[1, :, :], :])
        prediction1 = clf.predict(df_to_predict.loc[pd.IndexSlice[1, :, :], :])

        clf = svm.SVC(kernel='rbf', gamma='auto')
        clf.fit(df_features.loc[pd.IndexSlice[2:, :, :], :], df_labels.loc[pd.IndexSlice[2:, :, :], :])
        prediction2 = clf.predict(df_to_predict.loc[pd.IndexSlice[2:, :, :], :])

        prediction = np.zeros(len(prediction1)+len(prediction2), dtype=int)
        prediction[:len(prediction1)] = prediction1
        prediction[len(prediction1):] = prediction2

        self.exp.add_copy(
            old_name=distance_name, new_name=name_result,
            category='FoodCarrying', label='Is ant carrying?',
            description='Boolean giving if ants are carrying or not, for the ants next to the food'
        )
        self.exp.__dict__[name_result].df = self.exp.get_df(name_result).reindex(df_to_predict.index)
        self.exp.get_data_object(name_result).change_values(prediction)
        self.exp.write(name_result)

    def __get_df_to_predict_carrying(
            self, distance_diff_name, distance_name, orientation_name, speed_name):

        df_to_predict = self.exp.get_df(distance_name).join(self.exp.get_df(orientation_name), how='inner')
        self.exp.remove_object(orientation_name)
        df_to_predict = df_to_predict.join(self.exp.get_df(speed_name), how='inner')
        self.exp.remove_object(speed_name)
        df_to_predict = df_to_predict.join(self.exp.get_df(distance_diff_name), how='inner')
        self.exp.remove_object(distance_diff_name)
        df_to_predict.dropna(inplace=True)
        return df_to_predict

    def __get_training_features_and_labels4carrying(
            self, speed_name, orientation_name, distance_name, distance_diff_name, training_set_name):

        self.exp.filter_with_time_occurrences(
            name_to_filter=speed_name, filter_name=training_set_name,
            result_name='training_set_speed', replace=True)

        self.exp.filter_with_time_occurrences(
            name_to_filter=orientation_name, filter_name=training_set_name,
            result_name='training_set_orientation', replace=True)

        self.exp.filter_with_time_occurrences(
            name_to_filter=distance_name, filter_name=training_set_name,
            result_name='training_set_distance', replace=True)

        self.exp.filter_with_time_occurrences(
            name_to_filter=distance_diff_name, filter_name=training_set_name,
            result_name='training_set_distance_diff', replace=True)

        df_features = self.exp.training_set_distance.df.join(self.exp.training_set_orientation.df, how='inner')
        df_features = df_features.join(self.exp.training_set_speed.df, how='inner')
        df_features = df_features.join(self.exp.training_set_distance_diff.df, how='inner')
        df_features.dropna(inplace=True)

        self.exp.remove_object('training_set_speed')
        self.exp.remove_object('training_set_orientation')
        self.exp.remove_object('training_set_distance')
        self.exp.remove_object('training_set_distance_diff')

        df_labels = self.exp.get_df(training_set_name).reindex(df_features.index)

        return df_features, df_labels