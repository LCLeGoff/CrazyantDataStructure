import pandas as pd
import numpy as np

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.MiscellaneousTools.Geometry import angle_df, norm_angle_tab, norm_angle_tab2, norm_vect_df
from Tools.Plotter.Plotter import Plotter
from Tools.MiscellaneousTools.Geometry import distance
from shapely.geometry import LineString, Point


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

    def compute_mm10_distance2food_speed_next2food(self):
        name = 'mm10_distance2food_speed'
        res_name = name+'_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category=self.category, label='distance ant-food speed next to food',
            description='Instantaneous speed fo distance ant-food of ant next to food'
        )

        self.exp.write(res_name)

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

    def __reindexing(self, name_to_reindex, name2):
        id_exps = self.exp.get_df(name2).index.get_level_values(id_exp_name)
        id_ants = self.exp.get_df(name2).index.get_level_values(id_ant_name)
        frames = self.exp.get_df(name2).index.get_level_values(id_frame_name)
        idxs = pd.MultiIndex.from_tuples(list(zip(id_exps, frames)), names=[id_exp_name, id_frame_name])

        df = self.exp.get_df(name_to_reindex).copy()
        df = df.reindex(idxs)
        df[id_ant_name] = id_ants
        df.reset_index(inplace=True)
        df.columns = [id_exp_name, id_frame_name, 'x', 'y', id_ant_name]
        df.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)

        return df

    def compute_food_angular_component_ant_velocity(self, redo=False, redo_hist=False):
        result_name = 'food_angular_component_ant_velocity'
        print(result_name)
        hist_name = result_name+'_hist'

        dtheta = 0.1
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)

        if redo:
            food_name_x = 'mm10_food_x'
            food_name_y = 'mm10_food_y'
            food_name = 'food_xy'
            self.exp.load_as_2d(food_name_x, food_name_y, result_name=food_name, xname='x', yname='y', replace=True)

            food_speed_name_x = 'mm10_food_speed_x'
            food_speed_name_y = 'mm10_food_speed_y'
            food_speed_name = 'food_speed_xy'
            self.exp.load_as_2d(
                food_speed_name_x, food_speed_name_y, result_name=food_speed_name, xname='x', yname='y', replace=True)

            name_x = 'mm10_x'
            name_y = 'mm10_y'
            name_xy = 'xy'
            self.exp.load_as_2d(name_x, name_y, result_name=name_xy, xname='x', yname='y', replace=True)

            name_speed_x = 'mm10_speed_x'
            name_speed_y = 'mm10_speed_y'
            speed_name = 'speed_xy'
            self.exp.load_as_2d(name_speed_x, name_speed_y, result_name=speed_name, xname='x', yname='y', replace=True)

            df_food = self.__reindexing(food_name, name_xy)
            df_food_speed = self.__reindexing(food_speed_name, speed_name)

            df_food_ant = norm_vect_df(self.exp.get_df(name_xy)-df_food)
            df_velocity = norm_vect_df(self.exp.get_df(speed_name)-df_food_speed)

            df = angle_df(df_velocity, df_food_ant)

            self.exp.add_new1d_from_df(
                df=df, name=result_name, object_type='TimeSeries1d',
                category=self.category, label='Food angular component of the ant velocity',
                description='Angle between the food-ant vector and the vector,'
                            ' which  is ant velocity minus food velocity'
            )
            self.exp.write(result_name)

        self.compute_hist(name=result_name, bins=bins, hist_name=hist_name, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(normed=True, label='all', marker=None)
        plotter.draw_vertical_line(ax, np.pi/2.)
        plotter.draw_vertical_line(ax, -np.pi/2.)

        self.exp.load(['carrying', result_name])
        self.exp.filter_with_values(
            name_to_filter=result_name, filter_name='carrying', result_name='temp', replace=True)
        hist_name = self.exp.hist1d(name_to_hist='temp', replace=True, label='', description='')
        plotter2 = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        plotter2.plot(normed=True, preplot=(fig, ax), label='carrying', marker=None, c='w')

        ax.legend()
        plotter.save(fig)

    def compute_angle_velocity_food_next2food(self):
        name = 'angle_velocity_food'
        res_name = name + '_next2food'

        self.exp.load([name, 'is_xy_next2food'])

        self.exp.filter_with_values(
            name_to_filter=name, filter_name='is_xy_next2food', result_name=res_name,
            category=self.category, label='Food-ant velocity angle',
            description='Angle between the ant velocity and the ant-food center vector'
        )

        self.exp.write(res_name)

    def compute_attachment_xy(self):
        result_name = 'attachment_xy'
        print(result_name)

        name_x = 'mm10_x'
        name_y = 'mm10_y'
        name_xy = 'xy'
        self.exp.load_as_2d(name_x, name_y, name_xy, 'x', 'y')

        name_food_x = 'mm10_food_x'
        name_food_y = 'mm10_food_y'
        name_food_xy = 'food_xy'
        self.exp.load_as_2d(name_food_x, name_food_y, name_food_xy, 'x', 'y')

        name_orientation = 'mm10_orientation'
        name_radius = 'food_radius'
        self.exp.load([name_orientation, name_radius])

        self.exp.add_copy(old_name=name_xy, new_name=result_name, category=self.category,
                          label='Coordinates of the attachment point',
                          description='Coordinates of the intersection between the food boundary'
                                      ' and the line following the ant body '
                                      '(closest intersection of the ant position). When the ant is carrying, '
                                      'this point correspond to the attachment point '
                                      '(where the ant is attached to the food)')
        self.exp.get_df(result_name)[:] = np.nan

        def compute_attachment_point4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            id_ant = df.index.get_level_values(id_ant_name)[0]
            print(id_exp, id_ant)
            radius = self.exp.get_value(name_radius, id_exp)

            xys = df.loc[id_exp, id_ant, :]
            food_xys = self.exp.get_df(name_food_xy).loc[id_exp, :]
            food_xys = food_xys.reindex(xys.index).values

            orientations = self.exp.get_df(name_orientation).loc[id_exp, id_ant, :]
            orientations = orientations.loc[id_exp, id_ant, :]
            orientations = orientations.reindex(xys.index).values.ravel()

            xys = xys.values
            res = np.full((len(xys), 2), np.nan)

            alpha = np.tan(orientations)
            beta = xys[:, 1]-alpha*xys[:, 0]

            a = 1+alpha**2
            b = alpha*(beta-food_xys[:, 1])-food_xys[:, 0]
            c = food_xys[:, 0]**2+(beta-food_xys[:, 1])**2-radius**2

            delta = b**2-a*c
            mask = np.where(delta > 0)[0]
            x1 = (-b[mask]+np.sqrt(delta[mask]))/a[mask]
            x2 = (-b[mask]-np.sqrt(delta[mask]))/a[mask]
            y1 = alpha[mask]*x1+beta[mask]
            y2 = alpha[mask]*x2+beta[mask]
            pts1 = np.c_[x1, y1]
            pts2 = np.c_[x2, y2]

            dist1 = distance(xys[mask], pts1)
            dist2 = distance(xys[mask], pts2)
            dist_compare = dist1 < dist2

            mask_compare = np.where(dist_compare)[0]
            mask2 = mask[mask_compare]
            res[mask2, :] = pts1[mask_compare]

            mask_compare = np.where(~dist_compare)[0]
            mask2 = mask[mask_compare]
            res[mask2, :] = pts2[mask_compare]

            mask = np.where(delta == 0)[0]
            res[mask, 0] = -b[mask]/a[mask]
            res[mask, 1] = alpha[mask]*res[mask, 0]+beta[mask]

            self.exp.get_df(result_name).loc[id_exp, id_ant, :] = res

        self.exp.groupby(name_xy, [id_exp_name, id_ant_name], compute_attachment_point4each_group)

        self.exp.write(result_name)
