import numpy as np
import pandas as pd

from cv2 import cv2
from sklearn import svm

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.MiscellaneousTools.ArrayManipulation import auto_corr, get_entropy, get_interval_containing, \
    get_index_interval_containing
from Tools.Plotter.Plotter import Plotter
from Tools.Plotter.ColorObject import ColorObject


class AnalyseFoodCarrying(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'FoodCarrying'

    def compute_food_traj_length_around_first_attachment(self):
        before_result_name = 'food_traj_length_before_first_attachment'
        after_result_name = 'food_traj_length_after_first_attachment'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        food_traj_name = 'food_x'

        self.exp.load([food_traj_name, first_attachment_name, 'fps'])
        self.exp.add_new1d_empty(name=before_result_name, object_type='Characteristics1d', category=self.category,
                                 label='Food trajectory length before first outside attachment (s)',
                                 description='Length of the trajectory of the food in second '
                                             'before the first ant coming from outside attached to the food')
        self.exp.add_new1d_empty(name=after_result_name, object_type='Characteristics1d', category=self.category,
                                 label='Food trajectory length after first outside attachment (s)',
                                 description='Length of the trajectory of the food in second '
                                             'after the first ant coming from outside attached to the food')

        for id_exp in self.exp.id_exp_list:
            traj = self.exp.get_df(food_traj_name).loc[id_exp, :]
            frames = traj.index.get_level_values(id_frame_name)
            first_attachment_time = self.exp.get_value(first_attachment_name, id_exp)

            length_before = (first_attachment_time-int(frames.min()))/float(self.exp.fps.df.loc[id_exp])
            length_after = (int(frames.max())-first_attachment_time)/float(self.exp.fps.df.loc[id_exp])

            self.exp.change_value(name=before_result_name, idx=id_exp, value=length_before)
            self.exp.change_value(name=after_result_name, idx=id_exp, value=length_after)

        self.exp.write(before_result_name)
        self.exp.write(after_result_name)

        bins = range(0, 130, 15)
        hist_name = self.exp.hist1d(name_to_hist=before_result_name, bins=bins)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot()
        ax.grid()
        ax.set_xticks(range(0, 120, 15))
        plotter.save(fig)

        bins = range(-30, 500, 60)
        hist_name = self.exp.hist1d(name_to_hist=after_result_name, bins=bins)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot()
        ax.grid()
        ax.set_xticks(range(0, 430, 60))
        plotter.save(fig)

    def compute_non_outside_ant_attachment_frames(self):
        result_name = 'non_outside_ant_attachment_frames'
        carrying_name = 'non_outside_ant_carrying_intervals'

        self.__get_attachment_frames(carrying_name, result_name)

    def compute_outside_ant_attachment_frames(self):
        result_name = 'outside_ant_attachment_frames'
        carrying_name = 'outside_ant_carrying_intervals'

        self.__get_attachment_frames(carrying_name, result_name)

    def __get_attachment_frames(self, carrying_name, result_name):
        self.exp.load(carrying_name)
        nb_attach = len(self.exp.get_df(carrying_name))
        self.exp.get_data_object(carrying_name).df[:] = np.c_[range(nb_attach)]
        res = np.full((nb_attach, 3), -1)
        label = 'Attachment frames of outside ants'
        description = 'Frames when an ant from outside attached to the food, data is indexed by the experiment index' \
                      ' and the number of the attachment'

        def get_attachment_frame4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            frames = list(set(df.index.get_level_values(id_frame_name)))

            frames.sort()

            inters = np.array(df, dtype=int).ravel()
            lg = len(frames)

            res[inters[0]:inters[0] + lg, 0] = id_exp
            res[inters[0]:inters[0] + lg, 1] = range(1, lg + 1)
            res[inters[0]:inters[0] + lg, 2] = frames

            return df

        self.exp.groupby(carrying_name, id_exp_name, get_attachment_frame4each_group)
        res = res[res[:, -1] != -1, :]
        self.exp.add_new_dataset_from_array(array=res, name=result_name, index_names=[id_exp_name, 'th'],
                                            column_names=result_name, category=self.category,
                                            label=label, description=description)
        self.exp.write(result_name)

    def compute_outside_ant_attachment(self):
        carrying_name = 'carrying_intervals'
        outside_ant_name = 'from_outside'
        results_name = 'outside_ant_carrying_intervals'

        self.exp.load([carrying_name, outside_ant_name])
        self.exp.add_copy(old_name=carrying_name, new_name=results_name, category=self.category,
                          label='Carrying intervals of outside ants',
                          description='Intervals of the carrying periods for the ants coming from outside')

        def keep_only_outside_ants4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            id_ant = df.index.get_level_values(id_ant_name)[0]

            from_outside = self.exp.get_value(outside_ant_name, (id_exp, id_ant))
            if from_outside == 0:
                self.exp.get_df(results_name).drop(pd.IndexSlice[id_exp, id_ant], inplace=True)

        self.exp.get_df(results_name).groupby([id_exp_name, id_ant_name]).apply(keep_only_outside_ants4each_group)

        self.exp.write(results_name)

    def compute_non_outside_ant_attachment(self):
        carrying_name = 'carrying_intervals'
        outside_ant_name = 'from_outside'
        results_name = 'non_outside_ant_carrying_intervals'

        self.exp.load([carrying_name, outside_ant_name])
        self.exp.add_copy(old_name=carrying_name, new_name=results_name, category=self.category,
                          label='Carrying intervals of non outside ants',
                          description='Intervals of the carrying periods for the ants not coming from outside')

        def keep_only_non_outside_ants4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            id_ant = df.index.get_level_values(id_ant_name)[0]

            not_from_outside = 1-self.exp.get_value(outside_ant_name, (id_exp, id_ant))
            if not_from_outside == 0:
                self.exp.get_df(results_name).drop(pd.IndexSlice[id_exp, id_ant], inplace=True)

        self.exp.get_df(results_name).groupby([id_exp_name, id_ant_name]).apply(keep_only_non_outside_ants4each_group)

        self.exp.write(results_name)

    def compute_nbr_attachment_per_exp(self):
        outside_result_name = 'nbr_outside_attachment_per_exp'
        non_outside_result_name = 'nbr_non_outside_attachment_per_exp'
        result_name = 'nbr_attachment_per_exp'

        outside_attachment_name = 'outside_ant_carrying_intervals'
        non_outside_attachment_name = 'non_outside_ant_carrying_intervals'
        attachment_name = 'carrying_intervals'

        self.exp.load([attachment_name, non_outside_attachment_name, outside_attachment_name])

        self.exp.add_new1d_empty(name=outside_result_name, object_type='Characteristics1d', category=self.category,
                                 label='Number of outside attachments',
                                 description='Number of time an ant coming from outside attached to the food')

        self.exp.add_new1d_empty(name=non_outside_result_name, object_type='Characteristics1d', category=self.category,
                                 label='Number of non outside attachments',
                                 description='Number of time an ant not coming from outside attached to the food')

        self.exp.add_new1d_empty(name=result_name, object_type='Characteristics1d', category=self.category,
                                 label='Number of attachments',
                                 description='Number of time an ant attached to the food')

        for id_exp in self.exp.id_exp_list:
            attachments = self.exp.get_df(attachment_name).loc[id_exp, :]
            self.exp.change_value(name=result_name, idx=id_exp, value=len(attachments))

            attachments = self.exp.get_df(outside_attachment_name).loc[id_exp, :]
            self.exp.change_value(name=outside_result_name, idx=id_exp, value=len(attachments))

            attachments = self.exp.get_df(non_outside_attachment_name).loc[id_exp, :]
            self.exp.change_value(name=non_outside_result_name, idx=id_exp, value=len(attachments))

        self.exp.write(result_name)
        self.exp.write(outside_result_name)
        self.exp.write(non_outside_result_name)

        bins = np.arange(0, 600, 25)

        for name in [result_name, outside_result_name, non_outside_result_name]:
            hist_name = self.exp.hist1d(name_to_hist=name, bins=bins)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
            fig, ax = plotter.plot()
            ax.grid()
            plotter.save(fig)

    def compute_first_attachment_time_of_outside_ant(self):
        result_name = 'first_attachment_time_of_outside_ant'
        carrying_name = 'outside_ant_carrying_intervals'
        self.exp.load(carrying_name)

        self.exp.add_new1d_empty(name=result_name, object_type='Characteristics1d',
                                 category=self.category, label='First attachment time of a outside ant',
                                 description='First attachment time of an ant coming from outside')

        def compute_first_attachment4each_exp(df):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            frames = df.index.get_level_values(id_frame_name)
            print(id_exp)

            min_time = int(frames.min())
            self.exp.change_value(result_name, id_exp, min_time)

        self.exp.get_df(carrying_name).groupby(id_exp_name).apply(compute_first_attachment4each_exp)
        self.exp.get_data_object(result_name).df = self.exp.get_df(result_name).astype(int)
        self.exp.write(result_name)

    def compute_carrying_next2food_with_svm(self):
        name_result = 'carrying_next2food_from_svm'

        speed_name = 'speed_next2food'
        orientation_name = 'angle_body_food_next2food'
        distance_name = 'distance2food_next2food'
        distance_diff_name = 'distance2food_next2food_diff'
        training_set_name = 'carrying_training_set'

        self.exp.load(
            [training_set_name, speed_name, orientation_name, distance_name, distance_diff_name, 'mm2px'])

        self.exp.get_data_object(orientation_name).change_values(self.exp.get_df(orientation_name).abs())

        df_features, df_labels = self.__get_training_features_and_labels4carrying(
            orientation_name, distance_name, distance_diff_name, training_set_name)

        df_to_predict = self.__get_df_to_predict_carrying(
            distance_diff_name, distance_name, orientation_name)

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
            category=self.category, label='Is ant carrying?',
            description='Boolean giving if ants are carrying or not, for the ants next to the food (compute with svm)'
        )
        self.exp.__dict__[name_result].df = self.exp.get_df(name_result).reindex(df_to_predict.index)
        self.exp.get_data_object(name_result).change_values(prediction)
        self.exp.write(name_result)

    def __get_df_to_predict_carrying(
            self, distance_diff_name, distance_name, orientation_name):

        df_to_predict = self.exp.get_df(distance_name).join(self.exp.get_df(orientation_name), how='inner')
        self.exp.remove_object(orientation_name)
        df_to_predict = df_to_predict.join(self.exp.get_df(distance_diff_name), how='inner')
        self.exp.remove_object(distance_diff_name)
        df_to_predict.dropna(inplace=True)
        return df_to_predict

    def __get_training_features_and_labels4carrying(
            self, orientation_name, distance_name, distance_diff_name, training_set_name):

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
        df_features = df_features.join(self.exp.training_set_distance_diff.df, how='inner')
        df_features.dropna(inplace=True)

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

        def close_and_open4each_group(df):
            df_img = np.array(df, dtype=np.uint8)
            df_img = cv2.dilate(df_img, kernel=np.ones(dt, np.uint8))
            df_img = cv2.erode(df_img, kernel=np.ones(dt, np.uint8))
            df_img = cv2.erode(df_img, kernel=np.ones(dt, np.uint8))
            df_img = cv2.dilate(df_img, kernel=np.ones(dt, np.uint8))

            df[:] = df_img
            return df

        df_smooth = self.exp.get_df(name).groupby([id_exp_name, id_ant_name]).apply(close_and_open4each_group)

        self.exp.add_copy(old_name=name, new_name=result_name, category=self.category, label='Is ant carrying?',
                          description='Boolean giving if ants are carrying or not')
        self.exp.get_data_object(result_name).df = df_smooth

        self.exp.write(result_name)

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

        bins = np.arange(0.01, 1e2, 2 / 100.)
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
        hist_name = result_name+'_hist'

        bins = np.arange(0.01, 1e2, 2 / 100.)
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

    def compute_food_direction_error_evol_around_first_attachment(self, redo=False):
        name = 'food_direction_error'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        result_name = name+'_hist_evol_around_first_attachment'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)
        frame_intervals = np.arange(-2, 5., 1)*60*100

        result_label = 'Food direction error histogram evolution over time'
        result_description = 'Evolution over time of the histogram of food error direction,negative times (s)' \
                             ' correspond to periods before the first attachment of an outside ant'

        if redo:
            self.exp.load([name, first_attachment_name])

            new_times = 'new_times'
            self.exp.add_copy1d(name_to_copy=name, copy_name=new_times)
            self.exp.get_df(new_times).loc[:, new_times] = self.exp.get_index(new_times).get_level_values(id_frame_name)
            self.exp.operation_between_2names(name1=new_times, name2=first_attachment_name, func=lambda x, y: x - y)
            self.exp.get_df(new_times).reset_index(inplace=True)

            self.exp.get_df(name).reset_index(inplace=True)
            self.exp.get_df(name).loc[:, id_frame_name] = self.exp.get_df(new_times).loc[:, new_times]
            self.exp.get_df(name).set_index([id_exp_name, id_frame_name], inplace=True)

            self.exp.operation(name, lambda x: np.abs(x))

            self.exp.hist1d_time_evolution(name_to_hist=name, frame_intervals=frame_intervals, bins=bins,
                                           result_name=result_name, category=self.category,
                                           label=result_label, description=result_description)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\varphi$', ylabel='PDF', normed=True, label_suffix=' s')
        plotter.save(fig)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\varphi$', ylabel='PDF', label_suffix=' s')
        plotter.save(fig, suffix='non_norm')

    def compute_autocorrelation_food_phi(self, redo=False):

        name = 'food_phi'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        result_name = 'autocorrelation_'+name
        number_name = 'number'

        result_label = 'Food phi auto-correlation evolution over time'

        result_description = 'Auto-correlation of the angular coordinate of the food trajectory ' \
                             'at several intervals before and after the first  attachment' \
                             ' of an ant coming from outside, lags are in second'

        self.exp.load('fps')
        fps = int(self.exp.fps.df.iloc[0])
        if int(np.sum(self.exp.fps.df != fps)) == 0:

            dt = 0.5
            time_intervals = np.around(np.arange(-0.5, 2+2*dt, dt), 1)
            frame_intervals = np.array(time_intervals*fps*60, dtype=int)
            result_index_values = range(int(60*fps*dt)+1)

            if redo:
                self.__compute_autocorrelation(self.category, first_attachment_name, fps, frame_intervals, name,
                                               number_name, result_description, result_index_values,
                                               result_label, result_name, time_intervals)

            else:
                self.exp.load(result_name)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
            fig, ax = plotter.plot(xlabel='lag(s)', ylabel='', marker=None, label_suffix=' min')
            ax.set_xlim((0, 25))
            ax.set_ylim((-1, 1))
            ax.axhline(0, ls='--', c='k')
            ax.set_xticks(np.arange(0, 25, 5))
            ax.grid()
            plotter.save(fig)
        else:
            raise ValueError('fps not all the same')

    def compute_autocorrelation_food_velocity_phi(self, redo=False):

        name = 'food_velocity_phi'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        result_name = 'autocorrelation_'+name
        number_name = 'number'

        result_label = 'Food velocity phi auto-correlation evolution over time'

        result_description = 'Auto-correlation of the angular coordinate of the velocity of the food trajectory ' \
                             'at several intervals before and after the first  attachment' \
                             ' of an ant coming from outside, lags are in second'

        self.exp.load('fps')
        fps = int(self.exp.fps.df.iloc[0])
        if int(np.sum(self.exp.fps.df != fps)) == 0:

            dt = 0.5
            time_intervals = np.around(np.arange(-0.5, 2+2*dt, dt), 1)
            frame_intervals = np.array(time_intervals*fps*60, dtype=int)
            result_index_values = range(int(60*fps*dt)+1)

            if redo:
                self.__compute_autocorrelation(self.category, first_attachment_name, fps, frame_intervals, name,
                                               number_name, result_description, result_index_values, result_label,
                                               result_name, time_intervals)

            else:
                self.exp.load(result_name)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
            fig, ax = plotter.plot(xlabel='lag(s)', ylabel='', marker=None, label_suffix=' min')
            # ax.set_xlim((0, 25))
            ax.set_ylim((-0.25, 0.25))
            ax.axhline(0, ls='--', c='k')
            ax.set_xticks(np.arange(0, 25, 5))
            ax.grid()
            plotter.save(fig)
        else:
            raise ValueError('fps not all the same')

    def __compute_autocorrelation(self, category, first_attachment_name, fps, frame_intervals, name, number_name,
                                  result_description, result_index_values, result_label, result_name, time_intervals):
        self.exp.load([name, first_attachment_name])
        self.exp.add_new_empty_dataset(name=result_name, index_names='lag', column_names=time_intervals[:-1],
                                       index_values=result_index_values, fill_value=0, category=category,
                                       label=result_label, description=result_description)
        self.exp.add_new_empty_dataset(name=number_name, index_names='lag', column_names=time_intervals[:-1],
                                       index_values=result_index_values, fill_value=0, replace=True)
        for id_exp in self.exp.characteristic_timeseries_exp_frame_index:
            print(id_exp)
            food_phi = self.exp.get_df(name).loc[id_exp, :]
            first_attachment_time = self.exp.get_value(first_attachment_name, id_exp)
            for i in range(len(frame_intervals) - 1):
                frame0 = frame_intervals[i] + first_attachment_time
                frame1 = frame_intervals[i + 1] + first_attachment_time
                food_phi_temp = food_phi.loc[frame0:frame1]
                if len(food_phi_temp) != 0:
                    food_phi_temp.loc[:, name] = auto_corr(food_phi_temp)
                    food_phi_temp = food_phi_temp.iloc[:int(len(food_phi_temp) * 0.9), :]
                    frame_index = food_phi_temp.index.get_level_values(id_frame_name)
                    frame_index = frame_index - frame_index[0]
                    food_phi_temp.index = frame_index
                    self.exp.get_df(result_name).loc[frame_index, time_intervals[i]] \
                        += food_phi_temp.loc[frame_index, name]
                    self.exp.get_df(number_name).loc[frame_index, time_intervals[i]] += 1.
        self.exp.get_data_object(result_name).df /= self.exp.get_df(number_name)
        self.exp.get_data_object(result_name).df = self.exp.get_df(result_name).round(3)
        self.exp.get_df(result_name).index = self.exp.get_index(result_name) / fps
        self.exp.write(result_name)

    def __compute_autocorrelation_indiv(self, name, result_name, first_attachment_name, category, fps, dt,
                                        frame_intervals, time_intervals, result_label, result_description, redo):

        if redo:
            lag_list = range(0, int(60 * fps * dt) + 1, 10)
            result_index_values = np.array([(id_exp, lag) for id_exp in self.exp.id_exp_list for lag in lag_list])

            self.exp.load([name, first_attachment_name])
            self.exp.add_new_empty_dataset(name=result_name, index_names=[id_exp_name, 'lag'],
                                           column_names=time_intervals[:-1], index_values=result_index_values,
                                           category=category, label=result_label, description=result_description)

            for id_exp in self.exp.characteristic_timeseries_exp_frame_index:
                print(id_exp)

                data_df = self.exp.get_df(name).loc[id_exp, :]
                first_attachment_time = self.exp.get_value(first_attachment_name, id_exp)

                for i in range(len(frame_intervals) - 1):
                    frame0 = frame_intervals[i] + first_attachment_time
                    frame1 = frame_intervals[i + 1] + first_attachment_time

                    df_temp = data_df.loc[frame0:frame1]
                    if len(df_temp) != 0:
                        df_temp = df_temp.apply(auto_corr)
                        df_temp = df_temp.iloc[:int(len(df_temp) * 0.95), :]

                        frame_index = df_temp.index.get_level_values(id_frame_name)
                        frame_index -= frame_index[0]
                        df_temp.index = frame_index
                        df_temp = df_temp.reindex(lag_list)

                        index_slice = pd.IndexSlice[id_exp, :]
                        self.exp.get_df(result_name).loc[index_slice, time_intervals[i]] = np.array(df_temp)

            self.exp.get_data_object(result_name).df = self.exp.get_df(result_name).round(3)

            id_exp_index = np.array(self.exp.get_index_level_value(result_name, id_exp_name))
            lag_index = np.array(self.exp.get_index_level_value(result_name, 'lag'))/fps
            df_index = pd.MultiIndex.from_tuples(list(zip(id_exp_index, lag_index)), names=[id_exp_name, 'lag'])

            self.exp.get_df(result_name).index = df_index
            self.exp.write(result_name)

        else:
            self.exp.load(result_name)

    def compute_autocorrelation_food_velocity_phi_indiv(self, redo=False):

        name = 'food_velocity_phi'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        result_name = 'autocorrelation_'+name+'_indiv'

        result_label = 'Food velocity phi auto-correlation evolution over time'

        result_description = 'Auto-correlation of the angular coordinate of the velocity of the food trajectory ' \
                             'at several intervals before and after the first  attachment' \
                             ' of an ant coming from outside, lags are in second'

        self.exp.load('fps')
        fps = int(self.exp.fps.df.iloc[0])
        if int(np.sum(self.exp.fps.df != fps)) == 0:

            dt = 0.5
            time_intervals = np.around(np.arange(-0.5, 2+2*dt, dt), 1)
            frame_intervals = np.array(time_intervals*fps*60, dtype=int)

            self.__compute_autocorrelation_indiv(
                name=name, result_name=result_name, first_attachment_name=first_attachment_name, category=self.category,
                fps=fps, dt=dt, frame_intervals=frame_intervals, time_intervals=time_intervals,
                result_label=result_label, result_description=result_description, redo=redo)

            def plot_autocorrelation4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                df2 = df.set_index('lag')

                self.exp.add_new_dataset_from_df(df=df2, name='temp', category=self.category, replace=True)
                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))

                fig, ax = plotter.plot(xlabel='lag(s)', ylabel='', marker=None, label_suffix=' min')
                ax.axhline(0, ls='--', c='k')
                ax.grid()
                plotter.save(fig, name=id_exp, sub_folder=result_name)

            self.exp.groupby(result_name, id_exp_name, plot_autocorrelation4each_group)
        else:
            raise ValueError('fps not all the same')

    def compute_mm30s_dotproduct_food_velocity_exit_vs_food_velocity_vector_length(self):
        time = 30
        vel_length_name = 'mm'+str(time)+'s_food_velocity_vector_length'
        dotproduct_name = 'mm'+str(time)+'s_dotproduct_food_velocity_exit'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        self.exp.load([vel_length_name, dotproduct_name, first_attachment_name])

        result_name = 'information_trajectory'

        dt = 1.
        frame_intervals = np.around(np.arange(-2, 4, dt) * 60 * 100)
        colors = ColorObject.create_cmap('jet', frame_intervals[:-1])

        def plot2d(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            if id_exp >= 0:

                vel = df.loc[id_exp, :]
                dot_prod = self.exp.get_df(dotproduct_name).loc[id_exp, :]
                first_attachment_time = self.exp.get_value(first_attachment_name, id_exp)

                vel.index = vel.index.get_level_values(id_frame_name) - first_attachment_time
                dot_prod.index = dot_prod.index.get_level_values(id_frame_name) - first_attachment_time

                # vel /= vel.max()

                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(first_attachment_name))
                fig, ax = plotter.create_plot()
                for frame0, frame1 in zip(frame_intervals[:-1], frame_intervals[1:]):

                    vel2 = vel.loc[frame0:frame1]
                    dot_prod2 = dot_prod.loc[frame0:frame1]
                    dot_prod2.index = np.array(vel2).ravel()

                    self.exp.add_new_dataset_from_df(dot_prod2, name=id_exp, category=self.category, replace=True)

                    plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(id_exp))
                    fig, ax = plotter.plot(
                        xlabel='Velocity', ylabel='Dot product', c=colors[str(frame0)],
                        title_prefix='Exp ' + str(id_exp), preplot=(fig, ax),
                        label=r'$t\in$['+str(int(frame0/6000))+','+str(int(frame1/6000))+'[ min',
                        ls='', markeredgecolor=colors[str(frame0)])

                # ax.set_xlim((-0.05, 1.05))
                ax.set_ylim((-1.05, 1.05))
                ax.grid()
                ax.legend()
                plotter.save(fig, name=id_exp, sub_folder=result_name)

        self.exp.groupby(vel_length_name, id_exp_name, plot2d)

    def compute_mm30s_food_direction_error_vs_food_velocity_vector_length(self):
        time = 30
        vel_length_name = 'mm'+str(time)+'s_food_velocity_vector_length'
        error_name = 'mm'+str(time)+'s_food_direction_error'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        self.exp.load([vel_length_name, error_name, first_attachment_name])

        result_name = 'information_trajectory2'

        dt = 1.
        frame_intervals = np.around(np.arange(-2, 4, dt) * 60 * 100)
        colors = ColorObject.create_cmap('jet', frame_intervals[:-1])

        def plot2d(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            if id_exp >= 0:

                vel = df.loc[id_exp, :]
                error = self.exp.get_df(error_name).loc[id_exp, :]
                first_attachment_time = self.exp.get_value(first_attachment_name, id_exp)

                vel.index = vel.index.get_level_values(id_frame_name) - first_attachment_time
                error.index = error.index.get_level_values(id_frame_name) - first_attachment_time

                # vel /= vel.max()
                error = error.abs()

                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(first_attachment_name))
                fig, ax = plotter.create_plot()
                for frame0, frame1 in zip(frame_intervals[:-1], frame_intervals[1:]):

                    vel2 = vel.loc[frame0:frame1]
                    error2 = error.loc[frame0:frame1]
                    error2.index = np.array(vel2).ravel()

                    self.exp.add_new_dataset_from_df(error2, name=id_exp, category=self.category, replace=True)

                    plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(id_exp))
                    fig, ax = plotter.plot(
                        xlabel='Velocity', ylabel='Direction error', c=colors[str(frame0)],
                        title_prefix='Exp ' + str(id_exp), preplot=(fig, ax),
                        label=r'$t\in$['+str(int(frame0/6000))+','+str(int(frame1/6000))+'[ min',
                        ls='', markeredgecolor=colors[str(frame0)])

                # ax.set_xlim((-0.05, 1.05))
                ax.set_ylim((0, np.pi))
                ax.set_yticks(np.arange(0, np.pi+0.1, np.pi/4.))
                ax.grid()
                ax.legend()
                plotter.save(fig, name=id_exp, sub_folder=result_name)

        self.exp.groupby(vel_length_name, id_exp_name, plot2d)

    def compute_information_trajectory_around_attachment(self):
        time = 30
        vel_length_name = 'mm'+str(time)+'s_food_velocity_vector_length'
        error_name = 'mm'+str(time)+'s_food_direction_error'
        attachment_name = 'outside_ant_carrying_intervals'
        self.exp.load([vel_length_name, error_name, attachment_name])

        result_name = 'information_trajectory2_around_attachment'

        dt = np.array([-1, 5]) * 100

        def plot2d(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            print(id_exp)

            vel = df.loc[id_exp, :]
            attachment = self.exp.get_df(attachment_name).loc[id_exp, :]
            error = self.exp.get_df(error_name).loc[id_exp, :]

            error = error.abs()
            attachment_frames = np.array(attachment.index.get_level_values(id_frame_name))

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(attachment_name), category=self.category)
            fig, ax = plotter.create_plot()
            ax.plot(vel, error, c='w')

            for frame in attachment_frames:
                if frame in vel.index:
                    frame0 = frame+dt[0]
                    frame1 = frame+dt[1]

                    vel2 = vel.loc[frame0:frame1]
                    error2 = error.loc[frame0:frame1]
                    ax.plot(vel2, error2, c='blue')
                    ax.plot(vel2.loc[frame], error2.loc[frame], 'o', c='k')

            ax.set_xlabel('Velocity')
            ax.set_ylabel('Direction error')
            ax.set_ylim((0, np.pi))
            ax.set_yticks(np.arange(0, np.pi+0.1, np.pi/4.))
            ax.grid()
            plotter.save(fig, name=id_exp, sub_folder=result_name)

        self.exp.groupby(vel_length_name, id_exp_name, plot2d)

    def compute_mean_food_direction_error_around_outside_ant_attachments(self, redo=False):
        result_name = 'mean_food_direction_error_around_outside_ant_attachments'
        variable_name = 'mm1s_food_direction_error'
        attachment_name = 'outside_ant_carrying_intervals'

        column_name = ['average', '0', 'pi/8', 'pi/4', '3pi/8', 'pi/2', '5pi/8', '3pi/4', '7pi/8']
        val_intervals = np.arange(0, np.pi, np.pi / 8.)

        result_label = 'Food direction error mean around outside ant attachments'
        result_description = 'Mean of the food direction error during a time interval around a time an ant coming '\
                             'from outside attached to the food'

        self.__compute_mean_variable_around_attachments(result_name, variable_name, column_name, val_intervals,
                                                        attachment_name, self.category, result_label,
                                                        result_description, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(marker='')
        ax.axvline(0, ls='--', c='k')
        plotter.save(fig)

        fig, ax = plotter.plot(marker='')
        ax.axvline(0, ls='--', c='k')
        ax.set_xlim(-20, 20)
        plotter.save(fig, suffix='_zoom')

    def compute_mean_food_velocity_vector_length_around_outside_ant_attachments(self, redo=False):
        variable_name = 'mm1s_food_velocity_vector_length'
        attachment_name = 'outside_ant_carrying_intervals'
        result_name = 'mean_food_velocity_vector_length_around_outside_ant_attachments'

        val_intervals = range(0, 6)
        column_name = ['average']+[str(val) for val in val_intervals]

        result_label = 'Food velocity vector length mean around outside ant attachments'
        result_description = 'Mean of the food velocity vector length during a time interval around a time' \
                             ' an ant coming from outside attached to the food'

        self.__compute_mean_variable_around_attachments(result_name, variable_name, column_name, val_intervals,
                                                        attachment_name, self.category, result_label,
                                                        result_description, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot()
        ax.axvline(0, ls='--', c='k')
        plotter.save(fig)

        fig, ax = plotter.plot(marker='')
        ax.axvline(0, ls='--', c='k')
        ax.set_xlim(-20, 20)
        plotter.save(fig, suffix='_zoom')

    def __compute_mean_variable_around_attachments(self, result_name, variable_name, column_name, val_intervals,
                                                   attachment_name, category, result_label, result_description, redo):
        if redo:
            self.exp.load([variable_name, attachment_name, 'fps'])
            dt = 0.5
            times = np.arange(-20, 20 + dt, dt)
            self.exp.add_new_empty_dataset(name=result_name, index_names='time', column_names=column_name,
                                           index_values=times, fill_value=0, category=category,
                                           label=result_label,
                                           description=result_description)
            self.exp.add_new_empty_dataset(name='number', index_names='time', column_names=column_name,
                                           index_values=times, fill_value=0, replace=True)

            def compute_average4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                frame = df.index.get_level_values(id_frame_name)[0]

                print(id_exp, frame)
                fps = int(self.exp.get_value('fps', id_exp))

                df2 = self.exp.get_df(variable_name).loc[id_exp, :].abs()

                if frame in df2.index:
                    error0 = float(np.abs(self.exp.get_value(variable_name, (id_exp, frame))))

                    frame0 = int(frame + times[0] * fps)
                    frame1 = int(frame + times[-1] * fps)

                    var_df = df2.loc[frame0:frame1]
                    var_df.index -= frame
                    var_df.index /= fps
                    var_df = var_df.reindex(times)
                    var_df.dropna(inplace=True)

                    self.exp.get_df(result_name).loc[var_df.index, column_name[0]] += var_df[variable_name]
                    self.exp.get_df('number').loc[var_df.index, column_name[0]] += 1

                    i = get_index_interval_containing(error0, val_intervals)
                    self.exp.get_df(result_name).loc[var_df.index, column_name[i]] += var_df[variable_name]
                    self.exp.get_df('number').loc[var_df.index, column_name[i]] += 1

            self.exp.groupby(attachment_name, [id_exp_name, id_frame_name], compute_average4each_group)

            self.exp.get_data_object(result_name).df /= self.exp.get_df('number')

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

    def compute_mean_food_velocity_vector_length_vs_food_direction_error_around_outside_attachments(self, redo=False):
        vel_name = 'mm30s_food_velocity_vector_length'
        error_name = 'mm30s_food_direction_error'
        attachment_name = 'outside_ant_carrying_intervals'

        res_vel_name = 'mean_food_velocity_vector_length_vs_direction_error_around_outside_attachments_X'
        res_error_name = 'mean_food_velocity_vector_length_vs_direction_error_around_outside_attachments_Y'

        vel_intervals = [0, 2]
        error_intervals = np.around(np.arange(0, np.pi, np.pi / 2.), 3)
        dt = 1
        times = np.around(np.arange(-60, 60 + dt, dt), 1)
        index_values = np.array([(vel, error) for vel in vel_intervals for error in error_intervals])
        index_names = ['food_velocity_vector_length', 'food_direction_error']

        result_label_x = 'Food velocity vector length vs direction error around outside attachments (X)'
        result_description_x = 'X coordinates for the plot food velocity vector length' \
                               ' vs direction error around outside attachments'
        result_label_y = 'Food velocity vector length vs direction error around outside attachments (Y)'
        result_description_y = 'Y coordinates for the plot food velocity vector length' \
                               ' vs direction error around outside attachments)'

        if redo:
            self.exp.load([vel_name, error_name, attachment_name, 'fps'])

            self.exp.add_new_empty_dataset(name=res_vel_name, index_names=index_names,
                                           column_names=times, index_values=index_values, fill_value=0, replace=True,
                                           category=self.category, label=result_label_x,
                                           description=result_description_x)

            self.exp.add_new_empty_dataset(name=res_error_name, index_names=index_names,
                                           column_names=times, index_values=index_values, fill_value=0, replace=True,
                                           category=self.category, label=result_label_y,
                                           description=result_description_y)

            self.exp.add_new_empty_dataset(name='number', index_names=index_names,
                                           column_names=times, index_values=index_values, fill_value=0, replace=True)

            def compute_average4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                frame = df.index.get_level_values(id_frame_name)[0]

                print(id_exp, frame)
                fps = int(self.exp.get_value('fps', id_exp))

                df_vel = self.exp.get_df(vel_name).loc[id_exp, :].abs()
                df_error = self.exp.get_df(error_name).loc[id_exp, :].abs()

                if frame in df_vel.index:
                    error0 = float(np.abs(self.exp.get_value(error_name, (id_exp, frame))))
                    vel0 = float(np.abs(self.exp.get_value(vel_name, (id_exp, frame))))

                    frame0 = int(frame + times[0] * fps)
                    frame1 = int(frame + times[-1] * fps)

                    df_vel2 = df_vel.loc[frame0:frame1]
                    df_vel2.index -= frame
                    df_vel2.index /= fps
                    df_vel2 = df_vel2.reindex(times)
                    df_vel2.dropna(inplace=True)

                    df_error2 = df_error.loc[frame0:frame1]
                    df_error2.index -= frame
                    df_error2.index /= fps
                    df_error2 = df_error2.reindex(times)
                    df_error2.dropna(inplace=True)

                    i_error = get_interval_containing(error0, error_intervals)
                    i_vel = get_interval_containing(vel0, vel_intervals)

                    self.exp.get_df(res_vel_name).loc[(i_vel, i_error), df_vel2.index] += df_vel2[vel_name]
                    self.exp.get_df(res_error_name).loc[(i_vel, i_error), df_vel2.index] += df_error2[error_name]

                    self.exp.get_df('number').loc[(i_vel, i_error), df_vel2.index] += 1

            self.exp.groupby(attachment_name, [id_exp_name, id_frame_name], compute_average4each_group)

            self.exp.get_data_object(res_vel_name).df /= self.exp.get_df('number')
            self.exp.get_data_object(res_error_name).df /= self.exp.get_df('number')

            self.exp.get_data_object(res_vel_name).df = np.around(self.exp.get_data_object(res_vel_name).df, 3)
            self.exp.get_data_object(res_error_name).df = np.around(self.exp.get_data_object(res_error_name).df, 3)

            self.exp.write([res_vel_name, res_error_name])
        else:
            self.exp.load([res_vel_name, res_error_name])

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(res_vel_name))
        fig, ax = plotter.create_plot()
        colors = ColorObject.create_cmap('jet', self.exp.get_index(res_vel_name))
        for vel, error in self.exp.get_index(res_vel_name):

            c = colors[str((vel, error))]

            df2 = pd.DataFrame(np.array(self.exp.get_df(res_error_name).loc[(vel, error), :'0']),
                               index=np.array(self.exp.get_df(res_vel_name).loc[(vel, error), :'0']),
                               columns=['y'])
            self.exp.add_new_dataset_from_df(df=df2, name='temp', category=self.category, replace=True)
            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
            fig, ax = plotter.plot(xlabel='lag(s)', ylabel='', label_suffix=' min', marker='.', ls='',
                                   preplot=(fig, ax), c=c, markeredgecolor='w')

            df2 = pd.DataFrame(np.array(self.exp.get_df(res_error_name).loc[(vel, error), '0':]),
                               index=np.array(self.exp.get_df(res_vel_name).loc[(vel, error), '0':]),
                               columns=['y'])
            self.exp.add_new_dataset_from_df(df=df2, name='temp', category=self.category, replace=True)
            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
            fig, ax = plotter.plot(xlabel='lag(s)', ylabel='', label_suffix=' min', marker='.', ls='',
                                   preplot=(fig, ax), c=c, markeredgecolor='k')

            ax.plot(self.exp.get_df(res_vel_name).loc[(vel, error), '0'],
                    self.exp.get_df(res_error_name).loc[(vel, error), '0'], 'o', c=c)

        ax.set_xlim((0, 8))
        ax.set_ylim((0, np.pi))
        ax.set_xticks(vel_intervals)
        ax.set_yticks(error_intervals)
        ax.grid()
        plotter.save(fig, name='mean_food_velocity_vector_length_vs_food_direction_error_around_outside_attachments')
