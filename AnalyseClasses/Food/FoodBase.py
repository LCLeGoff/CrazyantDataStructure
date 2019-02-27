import numpy as np
import pandas as pd
from sklearn import svm

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Tools.MiscellaneousTools.Geometry import angle_df, norm_angle_tab, norm_angle_tab2


class AnalyseFoodBase:
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
        df_d = np.around(np.sqrt((df_f.x - self.exp.x.df.x) ** 2 + (df_f.y - self.exp.y.df.y) ** 2), 3)
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

    def compute_is_xy_next_to_food(self):
        name = 'is_xy_next_to_food'

        name_distance = 'distance_to_food'
        self.exp.load(name_distance)
        self.exp.add_copy1d(
            name_to_copy=name_distance, copy_name=name, category='FoodBase',
            label='Is next to food?', description='Is ants next to the food?'
        )

        neighbor_distance = 15.
        neighbor_distance2 = 5.
        self.exp.operation(name, lambda x: (x < neighbor_distance)*(x > neighbor_distance2))
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
            xname='x', yname='y', category='FoodBase',
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
            xname='x', yname='y', category='FoodBase',
            label='speed vector next to food', xlabel='x', ylabel='y', description='Speed vector of ants next to food'
        )

        self.exp.write(name)

    def compute_speed_next_to_food(self):
        name = 'speed'
        res_name = name+'_next_to_food'

        self.exp.load([name, 'is_xy_next_to_food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next_to_food', result_name=res_name,
            category='FoodBase', label='speed next to food', description='Instantaneous speed of ant next to food'
        )

        self.exp.write(res_name)

    def compute_mm5min_speed_next2food(self):
        name = 'speed_next_to_food'
        result_name = 'mm5min_speed_next2food'

        self.exp.load(name)
        time_window = 5*60*100

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of speed on 5min (' + str(time_window) + ' frames)',
            description='Moving mean of the instantaneous speed on a time window of 5min'
                        ' ('+str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm60s_speed_next2food(self):
        name = 'speed_next_to_food'
        result_name = 'mm60s_speed_next2food'

        self.exp.load(name)
        time_window = 6000.

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of speed on 1min (' + str(time_window) + ' frames)',
            description='Moving mean of the instantaneous speed on a time window of 1min'
                        ' ('+str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm10s_speed_next2food(self):
        name = 'speed_next_to_food'
        result_name = 'mm10s_speed_next2food'

        self.exp.load(name)
        time_window = 1000.

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of speed on 10s (' + str(time_window) + ' frames)',
            description='Moving mean of the instantaneous speed on a time window of 10s ('+str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm2s_speed_next2food(self):
        name = 'speed_next_to_food'
        result_name = 'mm2s_speed_next2food'

        self.exp.load(name)
        time_window = 200.

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of speed on 2s (' + str(time_window) + ' frames)',
            description='Moving mean of the instantaneous speed on a time window of 2s ('+str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm1s_speed_next2food(self):
        name = 'speed_next_to_food'
        result_name = 'mm1s_speed_next2food'

        self.exp.load(name)
        time_window = 100.

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of speed on 1s (' + str(time_window) + ' frames)',
            description='Moving mean of the instantaneous speed on a time window of 1s ('+str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm10_speed_next2food(self):
        name = 'speed_next_to_food'
        result_name = 'mm10_speed_next2food'

        self.exp.load(name)
        time_window = 10.

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name
        )

        self.exp.write(result_name)

    def compute_distance_to_food_next_to_food(self):
        name = 'distance_to_food'
        res_name = name+'_next_to_food'

        self.exp.load([name, 'is_xy_next_to_food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next_to_food', result_name=res_name,
            category='FoodBase', label='Food distance next to food',
            description='Distance between the food and the ants next to the food'
        )

        self.exp.write(res_name)

    def compute_mm5min_distance2food_next2food(self):
        name = 'distance_to_food_next_to_food'
        result_name = 'mm5min_distance2food_next2food'

        self.exp.load(name)
        time_window = 100*60*5

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of distance to food on 5min ('+str(time_window)+' frames)',
            description='Moving mean of the distance of the ants next to the food on a time window of 5min ('
                        + str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm60s_distance2food_next2food(self):
        name = 'distance_to_food_next_to_food'
        result_name = 'mm60s_distance2food_next2food'

        self.exp.load(name)
        time_window = 6000.

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of distance to food on 60s ('+str(time_window)+' frames)',
            description='Moving mean of the distance of the ants next to the food on a time window of 60s ('
                        + str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm10s_distance2food_next2food(self):
        name = 'distance_to_food_next_to_food'
        result_name = 'mm10s_distance2food_next2food'

        self.exp.load(name)
        time_window = 1000.

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of distance to food on 10s ('+str(time_window)+' frames)',
            description='Moving mean of the distance of the ants next to the food on a time window of 10s ('
                        + str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm1s_distance2food_next2food(self):
        name = 'distance_to_food_next_to_food'
        result_name = 'mm1s_distance2food_next2food'

        self.exp.load(name)
        time_window = 100.

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of distance to food on 1 s('+str(time_window)+' frames)',
            description='Moving mean of the distance of the ants next to the food on a time window of 1s ('
                        + str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm10_distance2food_next2food(self):
        name = 'distance_to_food_next_to_food'
        result_name = 'mm10_distance2food_next2food'

        self.exp.load(name)
        time_window = 10

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of distance to food on 10 '+str(time_window)+' frames',
            description='Moving mean of the distance of the ants next to the food on a time window of 10 '
                        + str(time_window) + ' frames'
        )

        self.exp.write(result_name)

    def compute_distance_to_food_next_to_food_differential(self):
        name = 'distance_to_food_next_to_food'
        result_name = 'distance_to_food_next_to_food_differential'

        self.exp.load([name, 'fps'])

        self.exp.add_copy(
            old_name=name, new_name=result_name, category='FoodBase', label='Food distance differential',
            description='Differential of the distance between the food and the ants', replace=True
        )

        def diff4each_group(df: pd.DataFrame):
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

        self.exp.distance_to_food_next_to_food_differential.df = \
            self.exp.get_df(result_name).groupby(['id_exp', 'id_ant']).apply(diff4each_group)

        self.exp.write(result_name)

    def compute_mm5min_distance2food_next2food_diff(self):
        name = 'distance_to_food_next_to_food_differential'
        result_name = 'mm5min_distance2food_next2food_diff'

        self.exp.load(name)
        time_window = 100*60*5

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of diff of distance to food on 5min ('+str(time_window)+' frames)',
            description='Moving mean of the differential of the the distance of the ants'
                        ' close to the food on a time window of 5min ('
                        + str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm60s_distance2food_next2food_diff(self):
        name = 'distance_to_food_next_to_food_differential'
        result_name = 'mm60s_distance2food_next2food_diff'

        self.exp.load(name)
        time_window = 6000.

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of diff of distance to food on 1min ('+str(time_window)+' frames)',
            description='Moving mean of the differential of the the distance of the ants'
                        ' close to the food on a time window of 1min ('
                        + str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm10s_distance2food_next2food_diff(self):
        name = 'distance_to_food_next_to_food_differential'
        result_name = 'mm10s_distance2food_next2food_diff'

        self.exp.load(name)
        time_window = 1000.

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of diff of distance to food on 10s ('+str(time_window)+' frames)',
            description='Moving mean of the differential of the the distance of the ants'
                        ' close to the food on a time window of 10s ('
                        + str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm2s_distance2food_next2food_diff(self):
        name = 'distance_to_food_next_to_food_differential'
        result_name = 'mm2s_distance2food_next2food_diff'

        self.exp.load(name)
        time_window = 200.

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of diff of distance to food on 2 s('+str(time_window)+' frames)',
            description='Moving mean of the differential of the the distance of the ants'
                        ' close to the food on a time window of 2s ('
                        + str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm1s_distance2food_next2food_diff(self):
        name = 'distance_to_food_next_to_food_differential'
        result_name = 'mm1s_distance2food_next2food_diff'

        self.exp.load(name)
        time_window = 100.

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of diff of distance to food on 1s ('+str(time_window)+' frames)',
            description='Moving mean of the differential of the the distance of the ants'
                        ' close to the food on a time window of 1s ('
                        + str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm10_distance2food_next2food_diff(self):
        name = 'distance_to_food_next_to_food_differential'
        result_name = 'mm10_distance2food_next2food_diff'

        self.exp.load(name)
        time_window = 10.

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of diff of distance to food on '+str(time_window)+' frames',
            description='Moving mean of the differential of the the distance of the ants'
                        ' close to the food on a time window of ' + str(time_window) + ' frames'
        )

        self.exp.write(result_name)

    def compute_mm3_distance2food_next2food_diff(self):
        name = 'distance_to_food_next_to_food_differential'
        time_window = 3
        result_name = 'mm'+str(int(time_window))+'_distance2food_next2food_diff'

        self.exp.load(name)

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of diff of distance to food on '+str(time_window)+' frames',
            description='Moving mean of the differential of the the distance of the ants'
                        ' close to the food on a time window of ' + str(time_window) + ' frames'
        )

        self.exp.write(result_name)

    def compute_orientation_next_to_food(self):
        name = 'orientation'
        res_name = name + '_next_to_food'

        self.exp.load([name, 'is_xy_next_to_food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next_to_food', result_name=res_name,
            category='FoodBase', label='orientation next to food', description='Body orientation of ant next to food'
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
            xname='x_ant', yname='y_ant', replace=True
        )
        df_food = self.__reindexing_food_xy(id_ants, idxs)

        df_ant_vector = df_food.copy()
        df_ant_vector.x = df_food.x - self.exp.xy_next_to_food.df.x
        df_ant_vector.y = df_food.y - self.exp.xy_next_to_food.df.y
        self.exp.add_copy('orientation_next_to_food', 'ant_food_orientation')
        self.exp.ant_food_orientation.change_values(angle_df(df_ant_vector))

        self.exp.add_copy('orientation_next_to_food', name)
        self.exp.rename(
            old_name=name, new_name=name, category='FoodBase', label='Body theta_res to food',
            description='Angle between the ant-food vector and the body theta_res vector'
        )
        self.exp.orientation_to_food.change_values(norm_angle_tab(
            self.exp.ant_food_orientation.df.ant_food_orientation
            - self.exp.orientation_next_to_food.df.orientation_next_to_food))
        self.exp.operation('orientation_to_food', lambda x: norm_angle_tab2(x))

        self.exp.write(name)

    def compute_mm5min_orientation2food(self):
        name = 'orientation_to_food'
        result_name = 'mm5min_orientation2food'

        self.exp.load(name)
        time_window = 5*60*100

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of diff of distance to food on 5min ('+str(time_window)+' frames)',
            description='Moving mean of the differential of the the distance of the ants'
                        ' close to the food on a time window of 5min ('
                        + str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm60s_orientation2food(self):
        name = 'orientation_to_food'
        result_name = 'mm60s_orientation2food'

        self.exp.load(name)
        time_window = 6000.

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of diff of distance to food on 60s ('+str(time_window)+' frames)',
            description='Moving mean of the differential of the the distance of the ants'
                        ' close to the food on a time window of 60s ('
                        + str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm10s_orientation2food(self):
        name = 'orientation_to_food'
        result_name = 'mm10s_orientation2food'

        self.exp.load(name)
        time_window = 1000.

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of diff of distance to food on 10s ('+str(time_window)+' frames)',
            description='Moving mean of the differential of the the distance of the ants'
                        ' close to the food on a time window of 10s ('
                        + str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm1s_orientation2food(self):
        name = 'orientation_to_food'
        result_name = 'mm1s_orientation2food'

        self.exp.load(name)
        time_window = 100.

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of diff of distance to food on 1s ('+str(time_window)+' frames)',
            description='Moving mean of the differential of the the distance of the ants'
                        ' close to the food on a time window of 1s ('
                        + str(time_window) + ' frames)'
        )

        self.exp.write(result_name)

    def compute_mm10_orientation2food(self):
        name = 'orientation_to_food'
        time_window = 10
        result_name = 'mm'+str(int(time_window))+'_orientation2food'

        self.exp.load(name)

        self.exp.moving_mean4frame_indexed_1d(
            name_to_average=name, time_window=time_window, result_name=result_name,
            label='MM of diff of distance to food on '+str(time_window)+' frames',
            description='Moving mean of the differential of the the distance of the ants'
                        ' close to the food on a time window of ' + str(time_window) + ' frames'
        )

        self.exp.write(result_name)

    def compute_is_carrying(self):
        name_result = 'is_carrying'
        speed_name = 'mm1s_speed_next2food'
        distance_name = 'distance_to_food_next_to_food'
        distance_diff_name = 'mm1s_distance2food_next2food_diff'
        training_set_name = 'carrying_training_set'
        self.exp.load([training_set_name, speed_name, distance_name, distance_diff_name])

        features, labels = self.__get_training_features_and_labels4carrying(
            speed_name, distance_name, distance_diff_name, training_set_name)

        mask = self.exp.get_df(distance_name).isna()[distance_name]\
            | self.exp.get_df(distance_diff_name).isna()[distance_diff_name]\
            | self.exp.get_df(speed_name).isna()[speed_name]

        speeds = np.array(self.exp.get_df(speed_name).mask(mask).dropna()[speed_name])
        distances = np.array(self.exp.get_df(distance_name).mask(mask).dropna()[distance_name])
        distance_diff = np.array(self.exp.get_df(distance_diff_name).mask(mask).dropna()[distance_diff_name])

        to_predict = np.array(list(zip(speeds, distances, distance_diff)))
        clf = svm.SVC(kernel='rbf')
        clf.fit(features, labels)
        prediction = clf.predict(to_predict)

        self.exp.add_copy(
            old_name=speed_name, new_name=name_result,
            category='FoodBase', label='Is ant carrying?',
            description='Boolean giving if ants are carrying or not'
        )
        self.exp.get_df(name_result).mask(mask).dropna()
        self.exp.get_data_object(name_result).change_values(prediction)
        self.exp.write(name_result)

    def __get_training_features_and_labels4carrying(
            self, speed_name, distance_name, distance_diff_name, training_set_name):

        self.exp.filter_with_time_occurrences(
            name_to_filter=speed_name, filter_name=training_set_name,
            result_name='training_set_speed', replace=True)
        self.exp.filter_with_time_occurrences(
            name_to_filter=distance_name, filter_name=training_set_name,
            result_name='training_set_distance', replace=True)
        self.exp.filter_with_time_occurrences(
            name_to_filter=distance_diff_name, filter_name=training_set_name,
            result_name='training_set_distance_diff', replace=True)

        speeds = np.abs(self.exp.training_set_speed.get_array())
        distances = self.exp.training_set_distance.get_array()
        distance_diff = self.exp.training_set_distance_diff.get_array()
        labels = np.array(self.exp.carrying_training_set.get_values())

        mask = np.where(~(np.isnan(distances))*~(np.isnan(distance_diff))*~(np.isnan(speeds)))[0]
        distance_diff = distance_diff[mask]
        distances = distances[mask]
        speeds = speeds[mask]
        labels = labels[mask]

        features = np.array(list(zip(speeds, distances, distance_diff)))
        return features, labels
