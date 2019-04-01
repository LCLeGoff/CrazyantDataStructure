import numpy as np
import pandas as pd
from cv2 import cv2
from sklearn import svm

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.MiscellaneousTools.ArrayManipulation import auto_corr, get_entropy
from Tools.Plotter.Plotter import Plotter


class AnalyseFoodCarrying(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)

    def compute_food_traj_length_around_first_attachment(self):
        before_result_name = 'food_traj_length_before_first_attachment'
        after_result_name = 'food_traj_length_after_first_attachment'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        food_traj_name = 'food_x'
        category = 'FoodCarrying'

        self.exp.load([food_traj_name, first_attachment_name, 'fps'])
        self.exp.add_new1d_empty(name=before_result_name, object_type='Characteristics1d', category=category,
                                 label='Food trajectory length before first outside attachment (s)',
                                 description='Length of the trajectory of the food in second '
                                             'before the first ant coming from outside attached to the food')
        self.exp.add_new1d_empty(name=after_result_name, object_type='Characteristics1d', category=category,
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

    def compute_outside_ant_attachment(self):
        carrying_name = 'carrying_intervals'
        outside_ant_name = 'from_outside'
        results_name = 'outside_ant_carrying_intervals'
        category = 'FoodCarrying'

        self.exp.load([carrying_name, outside_ant_name])
        self.exp.add_copy(old_name=carrying_name, new_name=results_name, category=category,
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
            category='FoodCarrying', label='Is ant carrying?',
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

    def compute_first_attachment_time_of_outside_ant(self):
        result_name = 'first_attachment_time_of_outside_ant'
        carrying_name = 'outside_ant_carrying_intervals'
        self.exp.load(carrying_name)

        self.exp.add_new1d_empty(name=result_name, object_type='Characteristics1d',
                                 category='FoodCarrying', label='First attachment time of a outside ant',
                                 description='First attachment time of an ant coming from outside')

        def compute_first_attachment4each_exp(df):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            frames = df.index.get_level_values(id_frame_name)

            min_time = int(frames.min())
            self.exp.change_value(result_name, id_exp, min_time)

        self.exp.get_df(carrying_name).groupby(id_exp_name).apply(compute_first_attachment4each_exp)
        self.exp.get_data_object(result_name).df = self.exp.get_df(result_name).astype(int)
        self.exp.write(result_name)

    def compute_food_phi_evol(self, redo=False):
        name = 'food_phi'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        result_name = 'food_phi_hist_evol'

        dtheta = np.pi/12.
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)
        frame_intervals = np.arange(-2, 6, 1)*60*100

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

            self.exp.hist1d_time_evolution(name_to_hist=name, frame_intervals=frame_intervals, bins=bins,
                                           result_name=result_name,  category='FoodCarrying',
                                           label='Food phi histogram evolution over time',
                                           description='Evolution overtime of the histogram of the angular coordinate '
                                                       'of the food trajectory, negative times (s) correspond '
                                                       'to periods before the first attachment of an outside ant')
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\varphi$', ylabel='PDF', normed=True, label_suffix=' s')
        plotter.save(fig)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\varphi$', ylabel='PDF', label_suffix=' s')
        plotter.save(fig, suffix='_non_norm')

    def compute_autocorrelation_food_phi(self, redo=False):

        name = 'food_phi'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        result_name = 'autocorrelation_food_phi'
        number_name = 'number'
        self.exp.load('fps')

        fps = int(self.exp.fps.df.iloc[0])
        if int(np.sum(self.exp.fps.df != fps)) == 0:

            dt = 0.5
            time_intervals = np.around(np.arange(-0.5, 2+2*dt, dt), 1)
            frame_intervals = np.array(time_intervals*fps*60, dtype=int)
            result_index_values = range(int(60*fps*dt)+1)

            if redo:
                self.exp.load([name, first_attachment_name])

                self.exp.add_new_empty_dataset(name=result_name, index_name='lag', column_names=time_intervals[:-1],
                                               index_values=result_index_values, fill_value=0, category='FoodCarrying',
                                               label='Food phi auto-correlation evolution over time',
                                               description='Auto-correlation of the angular coordinate'
                                                           ' of the food trajectory at several intervals'
                                                           ' before and after the first attachment'
                                                           ' of an ant coming from outside, lags are in second')

                self.exp.add_new_empty_dataset(name=number_name, index_name='lag', column_names=time_intervals[:-1],
                                               index_values=result_index_values, fill_value=0)

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
                            food_phi_temp = food_phi_temp.iloc[:int(len(food_phi_temp)*0.9), :]
                            frame_index = food_phi_temp.index.get_level_values(id_frame_name)
                            frame_index = frame_index - frame_index[0]
                            food_phi_temp.index = frame_index
                            self.exp.get_df(result_name).loc[frame_index, time_intervals[i]]\
                                += food_phi_temp.loc[frame_index, name]
                            self.exp.get_df(number_name).loc[frame_index, time_intervals[i]] += 1.

                self.exp.get_data_object(result_name).df /= self.exp.get_df(number_name)
                self.exp.get_data_object(result_name).df = self.exp.get_df(result_name).round(3)
                self.exp.get_df(result_name).index = self.exp.get_index(result_name)/fps

                self.exp.write(result_name)

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

    def compute_food_phi_entropy_evol_after_first_attachment(self, redo=False):

        food_phi_name = 'food_phi'
        result_name = 'food_phi_entropy_evol_after_first_attachment'
        first_attachment_name = 'first_attachment_time_of_outside_ant'

        dtheta = np.pi / 12.
        bins = np.around(np.arange(-np.pi + dtheta/2., np.pi + dtheta, dtheta), 3)
        dt = 0.25
        frame_intervals = np.around(np.arange(-2, 7, dt) * 60 * 100)
        max_entropy = np.log(2*np.pi*len(bins))

        self.__compute_entropy_evol(bins, 'FoodCarrying', first_attachment_name, food_phi_name, frame_intervals, redo,
                                    'Entropy of the angular coordinate distribution ', 'Food phi entropy over time',
                                    result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='time (s)')
        ax.set_xticks(range(-120, 8*60, 60))
        ax.grid()
        ax.axhline(max_entropy, ls='--', c='k', label='Max. entropy')
        ax.axvline(0, ls='--', c='k', label='t = 0')
        ax.legend(loc=0)
        plotter.save(fig)

        self.exp.operation(result_name, lambda x: max_entropy-x)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='time (s)')
        ax.set_xticks(range(-120, 8*60, 60))
        ax.grid()
        ax.axhline(max_entropy, ls='--', c='k', label='Max. entropy')
        ax.axvline(0, ls='--', c='k', label='t = 0')
        ax.legend(loc=0)
        plotter.save(fig, suffix='(info)')

    def compute_food_phi_entropy_evol_per_exp(self, redo=False):

        food_phi_name = 'food_phi'
        attachment_name = 'outside_ant_carrying_intervals'
        result_name = 'food_phi_entropy_evol_per_exp'
        category = 'FoodCarrying'

        result_label = 'Food phi entropy over time'
        result_description = 'Entropy of the angular coordinate distribution of the food trajectory ' \
                             'over time before and after the first attachment of an ant coming from outside, ' \
                             'negative times (s) correspond to periods before the first attachment of an outside ant'

        dtheta = np.pi / 8.
        bins = np.around(np.arange(-np.pi + dtheta/2., np.pi + dtheta, dtheta), 3)
        dt = 500
        frame_intervals = range(0, 10*60*100, dt)

        self.__compute_entropy_evol_per_exp(bins, category, food_phi_name, frame_intervals, redo, result_description,
                                            result_label, result_name)

        self.exp.load([food_phi_name, attachment_name])
        for id_exp in self.exp.id_exp_list:
            food_phi = self.exp.get_df(food_phi_name).loc[id_exp, :]
            frames = food_phi.index.get_level_values(id_frame_name)
            frame0 = int(frames.min())

            attachments = self.exp.get_df(attachment_name).loc[id_exp, :]
            attachments.reset_index(inplace=True)
            attachments = np.array(attachments)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name=id_exp)
            fig, ax = plotter.create_plot(figsize=(6, 4))

            # colors = plotter.color_object.create_cmap('hot_r', set(list(attachments[:, 0])))
            # for id_ant, frame, inter in attachments:
            #     ax.axvline((frame-frame0)/100, c=colors[str(id_ant)], alpha=0.5)

            ax.plot(np.array((frames-frame0)/100.), food_phi)
            plotter.plot(preplot=(fig, ax), figsize=(6, 4), xlabel='time (s)', ylabel='entropy')

            ax.legend(loc=1)
            plotter.save(fig, sub_folder=plotter.obj.name, name=id_exp)

    def compute_food_phi_speed_entropy_evol_per_exp(self, redo=False):

        food_phi_name = 'food_phi_speed'
        attachment_name = 'outside_ant_carrying_intervals'
        result_name = food_phi_name+'_entropy_evol_per_exp'
        category = 'FoodCarrying'

        result_label = 'Food phi speed entropy over time'
        result_description = 'Entropy of the speed of the angular coordinate distribution of the food trajectory ' \
                             'over time before and after the first attachment of an ant coming from outside, ' \
                             'negative times (s) correspond to periods before the first attachment of an outside ant'

        dtheta = np.pi / 8.
        bins = np.around(np.arange(-np.pi + dtheta / 2., np.pi + dtheta, dtheta), 3)
        dt = 500
        frame_intervals = range(0, 10 * 60 * 100, dt)

        self.__compute_entropy_evol_per_exp(bins, category, food_phi_name, frame_intervals, redo, result_description,
                                            result_label, result_name)

        self.exp.load([food_phi_name, attachment_name])
        for id_exp in self.exp.id_exp_list:
            food_phi = self.exp.get_df(food_phi_name).loc[id_exp, :]
            frames = food_phi.index.get_level_values(id_frame_name)
            frame0 = int(frames.min())

            attachments = self.exp.get_df(attachment_name).loc[id_exp, :]
            attachments.reset_index(inplace=True)
            attachments = np.array(attachments)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name=id_exp)
            fig, ax = plotter.create_plot(figsize=(6, 4))

            colors = plotter.color_object.create_cmap('hot_r', set(list(attachments[:, 0])))
            for id_ant, frame, inter in attachments:
                ax.axvline((frame-frame0)/100, c=colors[str(id_ant)], alpha=0.5)

            # ax.plot(np.array((frames - frame0) / 100.), food_phi)
            plotter.plot(preplot=(fig, ax), figsize=(6, 4), xlabel='time (s)', ylabel='entropy')

            ax.legend(loc=1)
            plotter.save(fig, sub_folder=plotter.obj.name, name=id_exp)

    def __compute_entropy_evol_per_exp(self, bins, category, food_phi_name, frame_intervals, redo, result_description,
                                       result_label, result_name):
        if redo:

            self.exp.load(food_phi_name)
            self.exp.add_new_empty_dataset(name=result_name, index_name='theta', column_names=self.exp.id_exp_list,
                                           index_values=np.array(frame_intervals) / 100, category=category,
                                           label=result_label,
                                           description=result_description)

            def get_entropy_evol4each_group(df: pd.DataFrame):
                id_exp0 = df.index.get_level_values(id_exp_name)[0]
                print(id_exp0)
                frames0 = df.index.get_level_values(id_frame_name)
                f0 = int(frames0.min())
                frames0 -= f0

                df.reset_index(inplace=True)
                df.loc[:, id_frame_name] = frames0
                df.set_index([id_exp_name, id_frame_name], inplace=True)

                food_phi_temp = 'food_phi_temp'
                self.exp.add_new1d_empty(name=food_phi_temp, object_type='CharacteristicTimeSeries1d', replace=True)
                self.exp.get_data_object(food_phi_temp).df = df

                food_phi_hist_name = self.exp.hist1d_time_evolution(name_to_hist=food_phi_temp,
                                                                    frame_intervals=frame_intervals, bins=bins,
                                                                    replace=True)

                self.exp.get_data_object(food_phi_hist_name).df /= np.sum(self.exp.get_df(food_phi_hist_name))
                # self.exp.get_data_object(food_phi_hist_name).df = self.exp.get_df(food_phi_hist_name).fillna(0)

                for i, t in enumerate(np.array(frame_intervals) / 100):
                    entropy = np.round(get_entropy(self.exp.get_df(food_phi_hist_name).loc[:, t]), 2)
                    self.exp.change_value(result_name, (t, id_exp0), entropy)

            self.exp.get_df(food_phi_name).groupby([id_exp_name]).apply(get_entropy_evol4each_group)

            self.exp.write(result_name)
            self.exp.remove_object(food_phi_name)
        else:
            self.exp.load(result_name)

    def compute_food_phi_speed_entropy_evol_after_first_attachment(self, redo=False):

        food_phi_name = 'food_phi_speed'
        result_name = food_phi_name+'_entropy_evol_after_first_attachment'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        category = 'FoodCarrying'

        result_label = 'Food phi entropy over time'
        result_description = 'Entropy of the angular coordinate distribution of the food trajectory' \
                             ' over time before and after the first attachment of an ant coming from outside,' \
                             ' negative times (s) correspond to periods before the first attachment of an outside ant'
        dtheta = np.pi / 12.
        bins = np.around(np.arange(-np.pi + dtheta/2., np.pi + dtheta, dtheta), 3)
        dt = 0.25
        frame_intervals = np.around(np.arange(-2, 7, dt) * 60 * 100)
        max_entropy = np.log(2*np.pi*len(bins))

        self.__compute_entropy_evol(bins, category, first_attachment_name, food_phi_name, frame_intervals, redo,
                                    result_description, result_label, result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='time (s)')
        ax.set_xticks(range(-120, 8*60, 60))
        ax.grid()
        ax.axhline(max_entropy, ls='--', c='k', label='Max. entropy')
        ax.axvline(0, ls='--', c='k', label='t = 0')
        ax.legend(loc=0)
        plotter.save(fig)

    def __compute_entropy_evol(self, bins, category, first_attachment_name, food_phi_name, frame_intervals, redo,
                               result_description, result_label, result_name):
        if redo:

            self.exp.load([food_phi_name, first_attachment_name])

            new_times = 'new_times'
            self.exp.add_copy1d(name_to_copy=food_phi_name, copy_name=new_times)
            self.exp.get_df(new_times).loc[:, new_times] = self.exp.get_index(new_times).get_level_values(
                id_frame_name)
            self.exp.operation_between_2names(name1=new_times, name2=first_attachment_name, func=lambda x, y: x - y)
            self.exp.get_df(new_times).reset_index(inplace=True)

            self.exp.get_df(food_phi_name).reset_index(inplace=True)
            self.exp.get_df(food_phi_name).loc[:, id_frame_name] = self.exp.get_df(new_times).loc[:, new_times]
            self.exp.get_df(food_phi_name).set_index([id_exp_name, id_frame_name], inplace=True)

            food_phi_hist_name = self.exp.hist1d_time_evolution(name_to_hist=food_phi_name,
                                                                frame_intervals=frame_intervals, bins=bins)

            self.exp.get_data_object(food_phi_hist_name).df /= np.sum(self.exp.get_df(food_phi_hist_name))
            self.exp.get_data_object(food_phi_hist_name).df = self.exp.get_df(food_phi_hist_name).fillna(0)

            times = np.array(self.exp.get_columns(food_phi_hist_name), dtype=float)

            self.exp.add_new_empty_dataset(name=result_name, index_name='theta', column_names='entropy',
                                           index_values=times, category=category,
                                           label=result_label, description=result_description)

            for t in times:
                entropy = np.round(get_entropy(self.exp.get_df(food_phi_hist_name).loc[:, t]), 2)
                self.exp.change_value(result_name, t, entropy)

            self.exp.write(result_name)

        else:
            self.exp.load(result_name)
