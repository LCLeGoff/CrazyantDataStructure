import numpy as np
import pandas as pd
from cv2 import cv2
from sklearn import svm

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from ExperimentGroups import ExperimentGroups
from Tools.Plotter.Plotter import Plotter


class AnalyseFoodCarrying:

    def __init__(self, root, group, exp: ExperimentGroups = None):
        if exp is None:
            self.exp = ExperimentGroupBuilder(root).build(group)
        else:
            self.exp = exp

    def compute_carrying_next2food_with_svm(self):
        name_result = 'carrying_next2food_from_svm'

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
        clf.fit(df_features.loc[pd.IndexSlice[:2, :, :], :], df_labels.loc[pd.IndexSlice[:2, :, :], :])
        prediction1 = clf.predict(df_to_predict.loc[pd.IndexSlice[:2, :, :], :])

        clf = svm.SVC(kernel='rbf', gamma='auto')
        clf.fit(df_features.loc[pd.IndexSlice[3:, :, :], :], df_labels.loc[pd.IndexSlice[3:, :, :], :])
        prediction2 = clf.predict(df_to_predict.loc[pd.IndexSlice[3:, :, :], :])

        prediction = np.zeros(len(prediction1)+len(prediction2), dtype=int)
        prediction[:len(prediction1)] = prediction1
        prediction[len(prediction1):] = prediction2

        self.exp.add_copy(
            old_name=distance_name, new_name=name_result,
            category='FoodCarrying', label='Is ant carrying?',
            description='Boolean giving if ants are carrying or not, for the ants next to the food (compute with svm)'
        )
        self.exp.__dict__[name_result].df = self.exp.get_df(name_result).reindex(df_to_predict.index)
        self.exp.get_data_object(name_result).change_values(prediction)
        self.exp.write(name_result)

    def __get_df_to_predict_carrying(
            self, distance_diff_name, distance_name, orientation_name, speed_name):

        df_to_predict = self.exp.get_df(distance_name).join(self.exp.get_df(orientation_name), how='inner')
        self.exp.remove_object(orientation_name)
        # df_to_predict = df_to_predict.join(self.exp.get_df(speed_name), how='inner')
        # self.exp.remove_object(speed_name)
        df_to_predict = df_to_predict.join(self.exp.get_df(distance_diff_name), how='inner')
        self.exp.remove_object(distance_diff_name)
        df_to_predict.dropna(inplace=True)
        return df_to_predict

    def __get_training_features_and_labels4carrying(
            self, speed_name, orientation_name, distance_name, distance_diff_name, training_set_name):

        # self.exp.filter_with_time_occurrences(
        #     name_to_filter=speed_name, filter_name=training_set_name,
        #     result_name='training_set_speed', replace=True)

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
        # df_features = df_features.join(self.exp.training_set_speed.df, how='inner')
        df_features = df_features.join(self.exp.training_set_distance_diff.df, how='inner')
        df_features.dropna(inplace=True)

        # self.exp.remove_object('training_set_speed')
        self.exp.remove_object('training_set_orientation')
        self.exp.remove_object('training_set_distance')
        self.exp.remove_object('training_set_distance_diff')

        df_labels = self.exp.get_df(training_set_name).reindex(df_features.index)

        return df_features, df_labels

    def compute_carrying_from_svm(self):
        name = 'carrying_next2food_from_svm'
        result_name = 'carrying_from_svm'
        self.exp.load([name, 'x'])

        self.exp.add_copy(old_name='x', new_name=result_name,
                          category='FoodCarrying', label='Is ant carrying?',
                          description='Boolean saying if the ant is carrying or not'
                          ' (data from svm closed and opened in terms of morphological transformation)')

        df = self.exp.get_reindexed_df(name_to_reindex=name, reindexer_name='x', fill_value=0)

        self.exp.get_data_object(result_name).df = df

        self.exp.write(result_name)

    def compute_carrying(self):
        name = 'carrying_from_svm'
        result_name = 'carrying'
        category = 'FoodCarrying'

        self.exp.load(name)
        dt = 11

        def close_and_open4each_group(df):
            df_img = np.array(df, dtype=np.uint8)
            df_img = cv2.dilate(df_img, kernel=np.ones(dt, np.uint8))
            df_img = cv2.erode(df_img, kernel=np.ones(dt, np.uint8))
            df_img = cv2.erode(df_img, kernel=np.ones(dt, np.uint8))
            df_img = cv2.dilate(df_img, kernel=np.ones(dt, np.uint8))

            df[:] = df_img
            return df

        df_smooth = self.exp.get_df(name).groupby([id_exp_name, id_ant_name]).apply(close_and_open4each_group)

        self.exp.add_copy(old_name=name, new_name=result_name, category=category, label='Is ant carrying?',
                          description='Boolean giving if ants are carrying or not')
        self.exp.get_data_object(result_name).df = df_smooth

        self.exp.write(result_name)

    def compute_carried_food(self):
        result_name = 'carried_food'
        category = 'FoodCarrying'
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
                                   category=category, label='Has the ant carried the food?',
                                   description='Boolean saying if the ant has at least once carried the food')

        self.exp.write(result_name)

    def compute_carrying_intervals(self, redo=False, redo_hist=False):

        result_name = 'carrying_intervals'
        hist_name = result_name+'_hist'

        bins = np.arange(0.01, 1e2, 2 / 100.)
        hist_label = 'Histogram of carrying time intervals'
        hist_description = 'Histogram of the time interval, while an ant is carrying the food'
        if redo is True:
            name = 'carrying'
            self.exp.load(name)

            self.exp.compute_time_intervals(name_to_intervals=name, category='FoodCarrying',
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
        hist_name = result_name+'_hist'

        bins = np.arange(0.01, 1e2, 2 / 100.)
        hist_label = 'Histogram of not carrying time intervals'
        hist_description = 'Histogram of the time intervals, while an ant is not carrying the food'
        if redo is True:
            name = 'carrying'
            self.exp.load(name)
            self.exp.get_data_object(name).df = 1-self.exp.get_data_object(name).df

            self.exp.compute_time_intervals(name_to_intervals=name, category='FoodCarrying',
                                            result_name=result_name, label='Not carrying time intervals',
                                            description='Time intervals, while an ant is not carrying the food (s)')

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
        fig, ax = plotter.plot(xlabel='Not carrying intervals', ylabel='PDF',
                               xscale='log', yscale='log', ls='', normed=True)
        plotter.save(fig)

    def compute_food_information_after_first_attachment(self):
        result_name = 'food_info_after_first_attachment'
        category = 'FoodCarrying'

        name = 'carrying_intervals'
        self.exp.load(['ant_from_outside', name, 'food_phi', 'fps'])
        if (self.exp.fps.df == self.exp.fps.df.iloc[0]).all():
            dt = np.array([-2, 8], dtype=int)*self.exp.fps.df.iloc[0]
            res = dict()
            for t in range(dt[0], dt[1]):
                res[t] = []

            def attachment4each_group(df):
                id_exp = df.index.get_index_value(id_exp_name)[0]
                id_ant = df.index.get_index_value(id_ant_name)[0]

                if self.exp.ant_from_outside.loc[id_exp, id_ant] == 1:
                    intervals = self.exp.get_df(name).loc[id_exp, id_ant, :]
                    if len(intervals) != 0:
                        dframes = dt*self.exp.fps.loc[id_exp]
                        frame_first_attachment = min(intervals.index.get_index_value(id_frame_name))
                        food_directions\
                            = self.exp.phi_food.loc[id_exp, id_ant, frame_first_attachment-dframes[0]
                                                                    :frame_first_attachment+dframes[1]+1]

                        for frame in food_directions.index.get_index_value(id_frame_name):
                            res[frame].append(food_directions.loc[id_exp, id_ant, frame])
                return df

        self.exp.get_data_object(name).df \
            = self.exp.get_df(name).groupby([id_exp_name, id_ant_name]).apply(attachment4each_group)

        df_res = self.exp.get_df(name).dropna()
        df_res.index = df_res.index.droplevel('frame')

        self.exp.add_new1d_from_df(df=df_res.astype(int), name=result_name, object_type='AntCharacteristics1d',
                                   category=category, label='Has the ant carried the food?',
                                   description='Boolean saying if the ant has at least once carried the food')

        self.exp.write(result_name)