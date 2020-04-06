import numpy as np
import pandas as pd

from cv2 import cv2
from sklearn import svm

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.MiscellaneousTools.ArrayManipulation import log_range
from Tools.MiscellaneousTools.Geometry import angle_df, angle_sum
from Tools.Plotter.Plotter import Plotter


class AnalyseFoodCarrying(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'FoodCarrying'

    def compute_outside_ant_carrying_intervals(self, redo=False, redo_hist=False):
        carrying_name = 'carrying_intervals'
        outside_ant_name = 'from_outside'
        result_name = 'outside_ant_carrying_intervals'

        bins = np.arange(2, 1e2, 0.5)
        hist_label = 'Histogram of carrying time intervals of outside ants'
        hist_description = 'Histogram of the time intervals, while an ant from outside is carrying the food'

        if redo:
            self.exp.load([carrying_name, outside_ant_name])
            self.exp.add_copy(old_name=carrying_name, new_name=result_name, category=self.category,
                              label='Carrying intervals of outside ants',
                              description='Intervals of the carrying periods for the ants coming from outside')

            def keep_only_outside_ants4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                id_ant = df.index.get_level_values(id_ant_name)[0]

                from_outside = self.exp.get_value(outside_ant_name, (id_exp, id_ant))
                if from_outside == 0:
                    self.exp.get_df(result_name).drop(pd.IndexSlice[id_exp, id_ant], inplace=True)

            self.exp.get_df(result_name).groupby([id_exp_name, id_ant_name]).apply(keep_only_outside_ants4each_group)

            self.exp.write(result_name)

        hist_name = self.compute_hist(
            name=result_name, bins=bins, redo=redo, redo_hist=redo_hist,
            hist_label=hist_label, hist_description=hist_description)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='Carrying intervals', ylabel='PDF',
                               xscale='log', yscale='log', ls='', normed=True)
        plotter.save(fig)

    def compute_inside_ant_carrying_intervals(self, redo=False, redo_hist=False):
        carrying_name = 'carrying_intervals'
        outside_ant_name = 'from_outside'
        result_name = 'inside_ant_carrying_intervals'

        bins = np.arange(2, 1e2, 0.5)
        hist_label = 'Histogram of carrying time intervals of inside ants'
        hist_description = 'Histogram of the time intervals, while an ant not from outside is carrying the food'

        if redo:
            self.exp.load([carrying_name, outside_ant_name])
            self.exp.add_copy(old_name=carrying_name, new_name=result_name, category=self.category,
                              label='Carrying intervals of inside ants',
                              description='Intervals of the carrying periods for the ants not coming from outside')

            def keep_only_non_outside_ants4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                id_ant = df.index.get_level_values(id_ant_name)[0]

                not_from_outside = 1-self.exp.get_value(outside_ant_name, (id_exp, id_ant))
                if not_from_outside == 0:
                    self.exp.get_df(result_name).drop(pd.IndexSlice[id_exp, id_ant], inplace=True)

            self.exp.get_df(result_name).groupby([id_exp_name, id_ant_name]).apply(
                keep_only_non_outside_ants4each_group)

            self.exp.write(result_name)

        hist_name = self.compute_hist(
            name=result_name, bins=bins, redo=redo, redo_hist=redo_hist,
            hist_label=hist_label, hist_description=hist_description)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='Carrying intervals', ylabel='PDF',
                               xscale='log', yscale='log', ls='', normed=True)
        plotter.save(fig)

    def compute_isolated_ant_carrying_intervals(self):
        attachment_name = 'attachment_intervals'
        result_name = 'isolated_ant_carrying_intervals'
        dt1 = 2
        dt2 = 2

        label = 'Isolated ant carrying intervals'
        description = 'Carrying intervals starting at a time' \
                      ' such that no other attachments occurred '+str(dt1)+'s before and '+str(dt2)+'s after'

        self.__isolated_attachments(dt1, dt2, attachment_name, description, label, result_name)

    def compute_isolated_outside_ant_carrying_intervals(self):
        attachment_name = 'outside_attachment_intervals'
        result_name = 'isolated_outside_ant_carrying_intervals'
        dt1 = 2
        dt2 = 2

        label = 'Isolated outside ant carrying intervals'
        description = 'Carrying intervals of outside ant starting at a time' \
                      ' such that no other attachments occurred '+str(dt1)+'s before and '+str(dt2)+'s after'

        self.__isolated_attachments(dt1, dt2, attachment_name, description, label, result_name)

    def compute_isolated_inside_ant_carrying_intervals(self):
        attachment_name = 'inside_attachment_intervals'
        result_name = 'isolated_non_outside_ant_carrying_intervals'
        dt1 = 2
        dt2 = 2

        label = 'Isolated non outside ant carrying intervals'
        description = 'Carrying intervals of non outside ant starting at a time' \
                      ' such that no other attachments occurred '+str(dt1)+'s before and '+str(dt2)+'s after'

        self.__isolated_attachments(dt1, dt2, attachment_name, description, label, result_name)

    def __isolated_attachments(self, dt1, dt2, attachment_name, description, label, result_name):
        self.exp.load(attachment_name)

        df_temp = self.exp.get_data_object(attachment_name).df

        def fct(df: pd.DataFrame):

            df2 = df.copy()
            df2['temp'] = np.inf
            df2['temp'].iloc[1:] = df.iloc[:-1].values.ravel()
            df2['res'] = (df[df.columns[0]] > dt2)*(df2['temp'] > dt1)

            df[~df2['res']] = np.nan

            return df

        df_temp = df_temp.groupby(id_exp_name).apply(fct)
        df_temp.dropna(inplace=True)

        self.exp.add_new1d_from_df(df=df_temp, name=result_name, object_type='Events1d',
                                   category=self.category, label=label, description=description)

        self.exp.write(result_name)

    def compute_carrying_next2food_with_svm(self):
        name_result = 'carrying_next2food_from_svm'

        speed_name = 'speed_next2food'
        orientation_name = 'angle_body_food_next2food'
        distance_name = 'distance2food_next2food'
        distance_diff_name = 'distance2food_next2food_diff'
        angle_velocity_food_name = 'food_angular_component_ant_velocity'
        list_feature_name = [speed_name, orientation_name, distance_name, distance_diff_name, angle_velocity_food_name]

        training_set_name = 'carrying_training_set'

        self.exp.load(list_feature_name + [training_set_name, 'mm2px'])

        self.exp.get_data_object(orientation_name).change_values(self.exp.get_df(orientation_name).abs())
        self.exp.get_data_object(angle_velocity_food_name).\
            change_values(self.exp.get_df(angle_velocity_food_name).abs())

        df = self.use_svm(list_feature_name, training_set_name)

        self.exp.add_new1d_from_df(
            df=df, name=name_result, object_type='TimeSeries1d', category=self.category, label='Is ant carrying?',
            description='Boolean giving if ants are carrying or not, for the ants next to the food (compute with svm)'
        )
        self.exp.write(name_result)

    def __get_df_to_predict_carrying(self, list_feature):

        df_to_predict = self.exp.get_df(list_feature[0])\

        for feature in list_feature[1:]:
            df_to_predict = df_to_predict.join(self.exp.get_df(feature), how='inner')
            self.exp.remove_object(feature)

        df_to_predict.dropna(inplace=True)
        return df_to_predict

    def __get_training_features_and_labels4carrying(
            self, list_features, training_set_name):

        self.exp.filter_with_time_occurrences(
            name_to_filter=list_features[0], filter_name=training_set_name,
            result_name='training_set_'+list_features[0], replace=True)
        df_features = self.exp.get_df('training_set_'+list_features[0])

        for feature in list_features[1:]:
            self.exp.filter_with_time_occurrences(
                name_to_filter=feature, filter_name=training_set_name,
                result_name='training_set_'+feature, replace=True)

            df_features = df_features.join(self.exp.get_df('training_set_'+feature), how='inner')

            self.exp.remove_object('training_set_'+feature)

        df_features.dropna(inplace=True)

        df_labels = self.exp.get_df(training_set_name).reindex(df_features.index)

        return df_features, df_labels

    def compute_food_rotation(self, redo=False, redo_hist=False):
        result_name = 'food_rotation'
        temp_name = 'temp'

        bins = np.arange(0, 3, 0.1)

        if redo:
            food_name_x = 'mm10_food_x'
            food_name_y = 'mm10_food_y'
            name_x = 'attachment_x'
            name_y = 'attachment_y'
            food_name = 'food_xy'
            name_xy = 'xy'
            self.exp.load_as_2d(name_x, name_y, result_name=name_xy, xname='x', yname='y', replace=True)
            self.exp.load_as_2d(food_name_x, food_name_y, result_name=food_name, xname='x', yname='y', replace=True)

            carrying_name = 'carrying_next2food_from_svm'
            self.exp.load([carrying_name, 'fps'])

            def erode4each_group(df):
                df_img = np.array(df, dtype=np.uint8)
                df_img = cv2.erode(df_img, kernel=np.ones(100, np.uint8))
                df[:] = df_img
                return df

            df_carrying = self.exp.get_df(carrying_name).groupby([id_exp_name, id_ant_name]).apply(erode4each_group)
            self.exp.change_df(carrying_name, df_carrying)

            self.exp.filter_with_values(
                name_to_filter=name_xy, filter_name=carrying_name, result_name=name_xy, replace=True)
            self.exp.remove_object(carrying_name)

            self.exp.add_new1d_from_df(self.exp.get_df(name_xy)['x'], name=temp_name, object_type='Events1d')

            def get_speed4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                id_ant = df.index.get_level_values(id_ant_name)[0]
                print(id_exp, id_ant)
                frames = np.array(df.index.get_level_values(id_frame_name))
                frame0 = frames[0]
                frame1 = frames[-1]

                df2 = df.loc[id_exp, id_ant, :]
                df2 -= self.exp.get_df(food_name).loc[id_exp, :]

                fps = self.exp.get_value('fps', id_exp)
                dframe = int(fps/2)
                vect1 = df2.loc[frame0+dframe:].copy()
                vect2 = df2.loc[:frame1-dframe].copy()

                dframe2 = int(dframe/2)
                vect1.index -= dframe2
                vect2.index += dframe2

                idx = self.exp.get_df(temp_name).loc[id_exp, id_ant, :].index.get_level_values(id_frame_name)
                vect1 = vect1.reindex(idx)
                vect2 = vect2.reindex(idx)

                df_angle = angle_df(vect1, vect2)

                df_angle = np.array(df_angle).ravel()
                df_angle /= dframe
                df_angle *= fps
                df_angle = np.around(df_angle, 6)

                self.exp.get_df(temp_name).loc[id_exp, id_ant, :] = np.c_[df_angle]

            self.exp.groupby(name_xy, [id_exp_name, id_ant_name], get_speed4each_group)

            self.exp.mean_over_exp_and_frames(name_to_average=temp_name, result_name=result_name,
                                              category=self.category, label='Food rotation',
                                              description='Rotation of the food (rad/s)', replace=True)
            self.exp.change_df(result_name, np.around(self.exp.get_df(result_name), 6))

            self.exp.write(result_name)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='Rotation (rad/s)', ylabel='PDF', ls='', normed=True)
        plotter.save(fig)

    def compute_mm10_food_rotation(self):
        name = 'food_rotation'
        time_window = 10

        self.exp.load(name)
        result_name = self.exp.rolling_mean(
            name_to_average=name, window=time_window, category=self.category, is_angle=True)

        self.exp.write(result_name)

    def compute_mm1s_food_rotation(self):
        name = 'food_rotation'
        time_window = 100

        self.exp.load(name)
        result_name = self.exp.rolling_mean(
            name_to_average=name, window=time_window, result_name='mm1s_'+name, category=self.category, is_angle=True)

        self.exp.write(result_name)

    def compute_ant_angular_speed(self):
        result_name = 'ant_angular_speed'
        print(result_name)

        food_rotation_name = 'mm10_food_rotation'
        self.exp.load([food_rotation_name, 'fps'])

        food_name_x = 'mm10_food_x'
        food_name_y = 'mm10_food_y'
        food_name = 'food_xy'
        self.exp.load_as_2d(food_name_x, food_name_y, result_name=food_name, xname='x', yname='y', replace=True)

        name_x = 'ant_body_end_x'
        name_y = 'ant_body_end_y'
        name_xy = 'xy'
        self.exp.load_as_2d(name_x, name_y, result_name=name_xy, xname='x', yname='y', replace=True)

        self.exp.add_copy(old_name=name_x, new_name=result_name, category=self.category,
                          label='Ant angular speed',
                          description='Ant angular speed relative to the food center')

        self.exp.get_df(result_name)[:] = np.nan

        def get_speed4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            id_ant = df.index.get_level_values(id_ant_name)[0]
            print(id_exp, id_ant)
            frames = np.array(df.index.get_level_values(id_frame_name))
            frame0 = frames[0]
            frame1 = frames[-1]

            df2 = df.loc[id_exp, id_ant, :]
            df2 -= self.exp.get_df(food_name).loc[id_exp, :]

            fps = self.exp.get_value('fps', id_exp)
            dframe = int(fps / 2)
            vect1 = df2.loc[frame0 + dframe:].copy()
            vect2 = df2.loc[:frame1 - dframe].copy()

            dframe2 = int(dframe / 2)
            vect1.index -= dframe2
            vect2.index += dframe2

            idx = df.index.get_level_values(id_frame_name)
            vect1 = vect1.reindex(idx)
            vect2 = vect2.reindex(idx)

            df_angle = angle_df(vect1, vect2)

            df_angle = np.array(df_angle).ravel()
            df_angle /= dframe
            df_angle *= fps
            df_angle = np.around(df_angle, 6)

            self.exp.get_df(result_name).loc[id_exp, id_ant, :] = np.c_[df_angle]

        self.exp.groupby(name_xy, [id_exp_name, id_ant_name], get_speed4each_group)

        self.exp.write(result_name)

    def compute_ant_food_relative_angular_speed(self):
        result_name = 'ant_food_relative_angular_speed'
        print(result_name)

        ant_angular_speed_name = 'ant_angular_speed'
        food_rotation_speed = 'food_rotation'
        self.exp.load([ant_angular_speed_name, food_rotation_speed, 'fps'])

        df_food = self.__reindexing(food_rotation_speed, ant_angular_speed_name, column_names=['temp'])

        self.exp.add_copy(ant_angular_speed_name, result_name, category=self.category,
                          label='Ant angular speed minus food rotation',
                          description='Difference between the ant angular speed and the food rotation')
        self.exp.get_data_object(result_name).df[result_name] -= df_food['temp']

        self.exp.write(result_name)

    def __reindexing(self, name_to_reindex, name2, column_names=None):
        if column_names is None:
            column_names = ['x', 'y']
        id_exps = self.exp.get_df(name2).index.get_level_values(id_exp_name)
        id_ants = self.exp.get_df(name2).index.get_level_values(id_ant_name)
        frames = self.exp.get_df(name2).index.get_level_values(id_frame_name)
        idxs = pd.MultiIndex.from_tuples(list(zip(id_exps, frames)), names=[id_exp_name, id_frame_name])

        df = self.exp.get_df(name_to_reindex).copy()
        df = df.reindex(idxs)
        df[id_ant_name] = id_ants
        df.reset_index(inplace=True)
        df.columns = [id_exp_name, id_frame_name]+column_names+[id_ant_name]
        df.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)

        return df

    def compute_carrying_next2food_with_svm_with_angular_speed(self):
        name_result = 'carrying_next2food_from_svm_with_angular_speed'

        orientation_name = 'angle_body_food_next2food'
        distance_name = 'distance2food_next2food'
        minus_angular_speed_name = 'ant_food_relative_angular_speed'
        list_feature_name = [orientation_name, distance_name, minus_angular_speed_name]

        training_set_name = 'carrying_training_set'

        self.exp.load(list_feature_name + [training_set_name, 'mm2px'])

        self.exp.operation(minus_angular_speed_name, np.abs)

        df = self.use_svm(list_feature_name, training_set_name)

        self.exp.add_new1d_from_df(
            df=df, name=name_result, object_type='TimeSeries1d', category=self.category, label='Is ant carrying?',
            description='Boolean giving if ants are carrying or not, for the ants next to the food. Using svm and '
                        'the angular speed gotten by a previous svm on carrying'
        )
        self.exp.write(name_result)

    def use_svm(self, list_feature_name, training_set_name):

        df_features, df_labels = self.__get_training_features_and_labels4carrying(list_feature_name, training_set_name)
        df_to_predict = self.__get_df_to_predict_carrying(list_feature_name)

        clf = svm.SVC(kernel='rbf', gamma='auto')
        clf.fit(df_features.loc[pd.IndexSlice[:2, :, :], :], df_labels.loc[pd.IndexSlice[:2, :, :], :])
        prediction1 = clf.predict(df_to_predict.loc[pd.IndexSlice[:2, :, :], :])

        clf = svm.SVC(kernel='rbf', gamma='auto')
        clf.fit(df_features.loc[pd.IndexSlice[3:, :, :], :], df_labels.loc[pd.IndexSlice[3:, :, :], :])
        prediction2 = clf.predict(df_to_predict.loc[pd.IndexSlice[3:, :, :], :])

        prediction = np.zeros(len(prediction1) + len(prediction2), dtype=int)
        prediction[:len(prediction1)] = prediction1
        prediction[len(prediction1):] = prediction2

        df = pd.DataFrame(prediction, index=df_to_predict.index)
        return df

    def compute_carrying_from_svm(self):
        name = 'carrying_next2food_from_svm'
        result_name = 'carrying_from_svm'
        self.exp.load([name, 'x'])

        self.exp.add_copy(old_name='x', new_name=result_name,
                          category=self.category, label='Is ant carrying?',
                          description='Boolean saying if the ant is carrying or not'
                          ' (data from svm closed and opened in terms of morphological transformation)')

        df = self.exp.get_reindexed_df(name_to_reindex=name, reindexer_name='x', fill_value=0)

        self.exp.get_data_object(result_name).df = df

        self.exp.write(result_name)

    def compute_carrying(self):
        name = 'carrying_from_svm'
        result_name = 'carrying'

        self.exp.load(name)
        dt = 11
        dt2 = 201

        def close_and_open4each_group(df):
            df_img = np.array(df, dtype=np.uint8)
            df_img = cv2.dilate(df_img, kernel=np.ones(dt, np.uint8))
            df_img = cv2.erode(df_img, kernel=np.ones(dt, np.uint8))

            df_img = cv2.erode(df_img, kernel=np.ones(dt2, np.uint8))
            df_img = cv2.dilate(df_img, kernel=np.ones(dt2, np.uint8))

            df[:] = df_img
            return df

        df_smooth = self.exp.get_df(name).groupby([id_exp_name, id_ant_name]).apply(close_and_open4each_group)

        self.exp.add_copy(old_name=name, new_name=result_name, category=self.category, label='Is ant carrying?',
                          description='Boolean giving if ants are carrying or not')
        self.exp.get_data_object(result_name).df = df_smooth

        self.exp.write(result_name)

    def compute_food_rotation_evol(self, redo=False):
        name = 'food_rotation'
        result_name = name + '_hist_evol'
        init_frame_name = 'food_first_frame'

        bins = np.arange(0, 3, 0.1)
        dx = 0.25
        start_frame_intervals = np.arange(0, 5., dx)*60*100
        end_frame_intervals = start_frame_intervals+dx*60*100*2

        if redo:
            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)
            self.exp.operation(name, lambda x: np.abs(x))
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                      end_frame_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category,
                                      label='Food rotation distribution over time (rad)',
                                      description='Histogram of the instantaneous rotation of the food over time '
                                                  ' (rad)')
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$v (mm/s)$', ylabel='PDF',
                               normed=True, label_suffix='s')
        plotter.save(fig)

    def compute_food_rotation_evol_around_first_outside_attachment(self, redo=False):
        name = 'food_rotation'
        result_name = name + '_hist_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        bins = np.arange(0, 3, 0.1)
        dx = 0.25
        start_frame_intervals = np.arange(-1, 4, dx)*60*100
        end_frame_intervals = start_frame_intervals+dx*60*100*2

        if redo:
            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)
            self.exp.operation(name, lambda x: np.abs(x))
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                      end_frame_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category,
                                      label='Food rotation distribution over time (rad)',
                                      description='Histogram of the instantaneous rotation of the food '
                                                  ' over time (rad)')
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$v (mm/s)$', ylabel='PDF',
                               normed=True, label_suffix='s')
        plotter.save(fig)

    def compute_food_rotation_variance_evol(self, redo=False):
        name = 'food_rotation'
        result_name = name + '_var_evol'
        init_frame_name = 'food_first_frame'

        dx = 0.1
        dx2 = 0.01
        start_frame_intervals = np.arange(0, 3.5, dx2)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        label = 'Variance of the food rotation distribution over time'
        description = 'Variance of the food rotation distribution over time'

        if redo:
            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            self.exp.variance_evolution(name_to_var=name, start_index_intervals=start_frame_intervals,
                                        end_index_intervals=end_frame_intervals,
                                        category=self.category, result_name=result_name,
                                        label=label, description=description)

            self.exp.write(result_name)
            self.exp.remove_object(name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(
            xlabel='Time (s)', ylabel=r'Variance $\sigma^2$',
            label_suffix='s', label=r'$\sigma^2$', title='', marker='')
        plotter.plot_smooth(50, c='orange', preplot=(fig, ax), label='smooth')
        plotter.plot_fit(typ='exp', preplot=(fig, ax), cst=[-.1, 0.05, 0.02], window=[50, 1000])
        plotter.save(fig)

    def compute_food_rotation_variance_evol_around_first_outside_attachment(self, redo=False):
        name = 'food_rotation'
        result_name = name + '_var_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.1
        dx2 = 0.01
        start_frame_intervals = np.arange(-1, 3.5, dx2)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        label = 'Variance of the food rotation distribution over time'
        description = 'Variance of the food rotation distribution over time'

        if redo:
            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            self.exp.variance_evolution(name_to_var=name, start_index_intervals=start_frame_intervals,
                                        end_index_intervals=end_frame_intervals,
                                        category=self.category, result_name=result_name,
                                        label=label, description=description)

            self.exp.write(result_name)
            self.exp.remove_object(name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(
            xlabel='Time (s)', ylabel=r'Variance $\sigma^2$',
            label_suffix='s', label=r'$\sigma^2$', title='', marker='')
        plotter.plot_smooth(50, c='orange', preplot=(fig, ax), label='smooth')
        plotter.plot_fit(typ='exp', preplot=(fig, ax), cst=(-1, 1, 1), window=[90, 1000])
        plotter.draw_vertical_line(ax)
        plotter.save(fig)

    def compute_food_spine_angle(self, redo=False):
        result_name = 'food_spine_angle'
        print(result_name)

        label = 'Food spine angle (rad)'
        description = 'Instantaneous angle of the food spine'

        if redo:
            name = 'food_rotation'
            self.exp.load([name, 'fps'])

            self.exp.add_copy1d(
                name_to_copy=name, copy_name=result_name, category=self.category, label=label, description=description)

            def get_speed4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                print(id_exp)

                fps = float(self.exp.get_value('fps', id_exp))
                df2 = np.array(df)/fps
                df2[0] = 0

                last_a = 0
                for i in range(1, len(df2)):
                    a2 = df2[i]
                    if ~np.isnan(a2):
                        df2[i] = angle_sum(last_a, -a2)[0]
                        last_a = a2

                # df2 = np.exp(1j*(df.copy()/fps*-1))
                # df2 = df2.cumsum()
                # df2 = pd.DataFrame(np.angle(df2), index=df2.index, columns=df2.columns)

                self.exp.get_df(result_name).loc[id_exp, :] = np.around(df2, 6)

            self.exp.groupby(name, id_exp_name, get_speed4each_group)
            self.exp.write(result_name)

    def compute_food_rotation_acceleration(self, redo=False, redo_hist=False):
        result_name = 'food_rotation_acceleration'
        print(result_name)

        hist_name = result_name + '_hist'
        bins = np.arange(0, 200, 0.25)
        label = 'Food rotation acceleration (mm/s^2)'
        description = 'Instantaneous acceleration of the food rotation (mm/s^2)'

        hist_label = 'Distribution of the '+label
        hist_description = 'Distribution of the '+description

        if redo:
            name = 'mm10_food_rotation'
            self.exp.load([name, 'fps'])

            self.exp.add_copy1d(
                name_to_copy=name, copy_name=result_name, category=self.category, label=label, description=description)

            def get_speed4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]

                dx = np.array(df)
                dx1 = dx[1, :].copy()
                dx2 = dx[-2, :].copy()
                dx[1:-1, :] = (dx[2:, :] - dx[:-2, :]) / 2.
                dx[0, :] = dx1 - dx[0, :]
                dx[-1, :] = dx[-1, :] - dx2

                dt = np.array(df.index.get_level_values(id_frame_name), dtype=float)
                dt.sort()
                dt[1:-1] = dt[2:] - dt[:-2]
                dt[0] = 1
                dt[-1] = 1
                dx[dt > 2] = np.nan

                fps = self.exp.get_value('fps', id_exp)
                self.exp.get_df(result_name).loc[id_exp, :] = np.around(dx*fps, 6)

            self.exp.groupby(name, id_exp_name, get_speed4each_group)
            self.exp.write(result_name)

        self.compute_hist(name=result_name, bins=bins, hist_name=hist_name, hist_label=hist_label,
                          hist_description=hist_description, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel=r'$dv/dt$ (mm/s^2)', ylabel='PDF',
                               normed=True, label_suffix='s')
        # ax.set_xlim((0, 20))
        plotter.save(fig)

    def compute_carried_food(self):
        result_name = 'carried_food'
        name = 'carrying'

        self.exp.load(name)

        def carried_food4each_group(carrying):
            xys = np.array(carrying)
            carrying[:] = np.nan
            carried_food = any(xys)
            carrying.iloc[0, :] = carried_food

            return carrying

        self.exp.get_data_object(name).df\
            = self.exp.get_df(name).groupby(['id_exp', 'id_ant']).apply(carried_food4each_group)

        df_res = self.exp.get_df(name).dropna()
        df_res.index = df_res.index.droplevel('frame')

        self.exp.add_new1d_from_df(df=df_res.astype(int), name=result_name, object_type='AntCharacteristics1d',
                                   category=self.category, label='Has the ant carried the food?',
                                   description='Boolean saying if the ant has at least once carried the food')

        self.exp.write(result_name)

    def compute_carrying_intervals(self, redo=False, redo_hist=False):

        result_name = 'carrying_intervals'
        hist_name = result_name+'_hist'

        bins = np.arange(2, 1e2, 0.5)
        hist_label = 'Histogram of carrying time intervals'
        hist_description = 'Histogram of the time interval, while an ant is carrying the food'
        if redo is True:
            name = 'carrying'
            self.exp.load(name)

            self.exp.compute_time_intervals(name_to_intervals=name, category=self.category,
                                            result_name=result_name, label='Carrying time intervals',
                                            description='Time intervals during which ants are carrying (s)')

            self.exp.write(result_name)

            self.exp.hist1d(name_to_hist=result_name, bins=bins, label=hist_label, description=hist_description)
            self.exp.write(hist_name)

        elif redo_hist is True:
            self.exp.load(result_name)
            self.exp.hist1d(name_to_hist=result_name, bins=bins, label=hist_label, description=hist_description)
            self.exp.write(hist_name)

        else:
            self.exp.load(hist_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='Carrying intervals', ylabel='PDF',
                               xscale='log', yscale='log', ls='', normed=True)
        plotter.save(fig)

    def compute_not_carrying_intervals(self, redo=False, redo_hist=False):

        result_name = 'not_carrying_intervals'

        bins = np.arange(2, 1e2, 0.5)
        hist_label = 'Histogram of not carrying time intervals'
        hist_description = 'Histogram of the time intervals, while an ant is not carrying the food'
        if redo is True:
            name = 'carrying'
            self.exp.load(name)
            self.exp.get_data_object(name).df = 1-self.exp.get_data_object(name).df

            self.exp.compute_time_intervals(name_to_intervals=name, category=self.category,
                                            result_name=result_name, label='Not carrying time intervals',
                                            description='Time intervals, while an ant is not carrying the food (s)')

            self.exp.write(result_name)
            self.exp.remove_object(name)

        hist_name = self.compute_hist(name=result_name, bins=bins, hist_label=hist_label,
                                      hist_description=hist_description, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='Not carrying intervals', ylabel='PDF',
                               xscale='log', yscale='log', ls='', normed=True)
        plotter.save(fig)

    def compute_aviram_carrying_intervals(self, redo=False):

        result_name = 'aviram_carrying_intervals'
        hist_name = result_name+'_hist'

        bins = log_range(2, 300, 20)
        label = 'Carrying time intervals'
        description = 'Time intervals during which ants are carrying (s) of the nature communication paper published' \
                      ' by Aviram Gelblum et al., 2015'
        hist_label = 'Histogram of '+label.lower()
        hist_description = 'Histogram of '+description
        if redo is True:
            df = pd.read_csv(self.exp.root+'Carrying_Durations_Aviram.csv')
            df.index.name = 'id'
            self.exp.add_new_dataset_from_df(df=df, name=result_name, category=self.category,
                                             label=label, description=description)
            self.exp.write(result_name)

            self.exp.hist1d(name_to_hist=result_name, bins=bins, label=hist_label, description=hist_description)
            self.exp.write(hist_name)

        else:
            self.exp.load(hist_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.create_plot()

        carrying_name = 'carrying_intervals_hist'
        self.exp.load(carrying_name)
        plotter2 = Plotter(root=self.exp.root, obj=self.exp.get_data_object(carrying_name))
        plotter2.plot(preplot=(fig, ax), xlabel='Carrying intervals',
                      ylabel='PDF', xscale='log', yscale='log', ls='', normed=True)

        plotter.plot(preplot=(fig, ax), c='w',
                     xlabel='Carrying intervals', ylabel='PDF', xscale='log', yscale='log', normed=True)
        plotter.save(fig)

    def compute_nb_carriers(self, redo=False, redo_hist=False):

        result_name = 'nb_carriers'

        bins = range(20)

        if redo:
            carrying_name = 'carrying'
            food_name = 'food_x'
            self.exp.load([carrying_name, food_name])

            self.exp.sum_over_exp_and_frames(name_to_average=carrying_name, result_name=result_name,
                                             category=self.category, label='Number of carriers',
                                             description='Number of ants carrying the food')
            self.exp.get_data_object(result_name).df = self.exp.get_df(result_name).reindex(
                self.exp.get_index(food_name), fill_value=0)
            self.exp.get_data_object(result_name).df = self.exp.get_df(result_name).astype(int)

            self.exp.write(result_name)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='Number of carriers', ylabel='PDF', ls='', normed=True)
        plotter.save(fig)

    def compute_nb_carriers_evol_around_first_outside_attachment(self, redo=False):
        name = 'nb_carriers'
        result_name = name + '_hist_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        bins = np.arange(0, 21)-0.5
        start_frame_intervals = np.array([-2, 0])*60*100
        end_frame_intervals = np.array([0, 5])*60*100

        if redo:
            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                      end_frame_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category,
                                      label='Food rotation distribution over time (rad)',
                                      description='Histogram of the instantaneous rotation of the food '
                                                  ' over time (rad)')
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='nb of carriers', ylabel='PDF',
                               normed=True, label_suffix='s')
        ax.set_xticks(range(0, 21, 2))
        ax.grid()
        plotter.save(fig)

    def compute_nb_outside_carriers(self, redo=False, redo_hist=False):

        result_name = 'nb_outside_carriers'

        bins = range(20)

        if redo:
            carrying_name = 'carrying'
            food_name = 'food_x'
            from_outside_name = 'from_outside'
            self.exp.load([carrying_name, food_name, from_outside_name])

            def is_from_outside4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                id_ant = df.index.get_level_values(id_ant_name)[0]

                from_outside = self.exp.get_value(from_outside_name, (id_exp, id_ant))

                df[:] *= from_outside
                return df

            self.exp.groupby(carrying_name, [id_exp_name, id_ant_name], is_from_outside4each_group)

            self.exp.sum_over_exp_and_frames(name_to_average=carrying_name, result_name=result_name,
                                             category=self.category, label='Number of outside carriers',
                                             description='Number of ants from outside carrying the food', replace=True)

            self.exp.get_data_object(result_name).df = self.exp.get_df(result_name).reindex(
                self.exp.get_index(food_name), fill_value=0)
            self.exp.get_data_object(result_name).df = self.exp.get_df(result_name).astype(int)

            self.exp.write(result_name)
            self.exp.remove_object(carrying_name)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='Number of outside carriers', ylabel='PDF', ls='', normed=True)
        plotter.save(fig)

    def compute_food_rotation_vs_nb_carriers(self, redo=False):

        xname = 'nb_carriers'
        yname = 'food_rotation'
        result_name = yname+'_vs_'+xname
        if redo:
            self.exp.load([xname, yname])
            self.exp.operation(yname, np.abs)
            result_name = self.exp.vs(xname, yname, n_bins=range(20), x_are_integers=True)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot_with_error(ms=10)
        plotter.save(fig)

    def compute_food_speed_vs_nb_carriers(self, redo=False):

        xname = 'nb_carriers'
        yname = 'food_speed'
        result_name = yname+'_vs_'+xname
        if redo:
            self.exp.load([xname, yname])

            result_name = self.exp.vs(xname, yname, n_bins=range(20), x_are_integers=True)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot_with_error(ms=10)
        plotter.save(fig)

    def compute_nb_carriers_mean_evol(self, redo=False):

        name = 'nb_carriers'
        result_name = name + '_mean_evol'
        init_frame_name = 'food_first_frame'

        dx = 0.01
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(0, 4., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Mean of the number of carriers over time'
        description = 'Mean of the number of ants carrying the food over time'

        if redo:

            self._mean_evol_for_nb_carrier(name, result_name, init_frame_name, start_frame_intervals,
                                           end_frame_intervals, label, description)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Mean', label_suffix='s', label='Mean', marker='')
        ax.set_ylim((0, 10))
        plotter.save(fig)

    def _mean_evol_for_nb_carrier(self, name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                  label, description):
        self.exp.load(name)

        self.change_first_frame(name, init_frame_name)

        self.exp.mean_evolution(name_to_var=name, start_index_intervals=start_frame_intervals,
                                end_index_intervals=end_frame_intervals,
                                category=self.category, result_name=result_name,
                                label=label, description=description, replace=True)
        self.exp.write(result_name)
        self.exp.remove_object(name)

    def compute_nb_outside_carriers_mean_evol(self, redo=False):

        name = 'nb_outside_carriers'
        result_name = name + '_mean_evol'
        init_frame_name = 'food_first_frame'

        dx = 0.01
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(0, 4., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Mean of the number of outside carriers over time'
        description = 'Mean of the number of ants coming from outside carrying the food over time'

        if redo:

            self._mean_evol_for_nb_carrier(name, result_name, init_frame_name, start_frame_intervals,
                                           end_frame_intervals, label, description)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Mean', label_suffix='s', label='Mean', marker='')
        plotter.save(fig)

    def compute_nb_carriers_mean_evol_around_first_attachment(self, redo=False):

        name = 'nb_carriers'
        result_name = name + '_mean_evol_around_first_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.01
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(-2, 4., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Mean of the number of carries over time'
        description = 'Mean of the number of ants carrying the food over time'

        if redo:
            self._mean_evol_for_nb_carrier(name, result_name, init_frame_name, start_frame_intervals,
                                           end_frame_intervals, label, description)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Mean', label_suffix='s', label='Mean', marker='')
        plotter.draw_vertical_line(ax, 0, label='first attachment')
        ax.legend()
        plotter.save(fig)
