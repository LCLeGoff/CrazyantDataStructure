import numpy as np
import pandas as pd

from cv2 import cv2
from sklearn import svm

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.MiscellaneousTools.ArrayManipulation import get_interval_containing, get_index_interval_containing, log_range
from Tools.MiscellaneousTools.Geometry import angle_df, angle_sum
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
        res = np.full((nb_attach, 3), -1)
        label = 'Attachment frames of outside ants'
        description = 'Frames when an ant from outside attached to the food, data is indexed by the experiment index' \
                      ' and the number of the attachment'

        def get_attachment_frame4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]

            df2 = df.reset_index()
            df2.drop(columns=[id_exp_name, id_ant_name], inplace=True)
            df2.set_index(id_frame_name, inplace=True)
            df2 = df2[~df2.index.duplicated()]

            frames = list(df2.index.get_level_values(id_frame_name))
            frames.sort()

            inters = np.array(df2).ravel()
            mask = np.where(inters > 1)[0]
            frames = np.array(frames)[mask]
            lg = len(frames)

            inter = np.where(res[:, 0] == -1)[0][0]
            res[inter:inter + lg, 0] = id_exp
            res[inter:inter + lg, 1] = range(1, lg + 1)
            res[inter:inter + lg, 2] = frames

            return df

        self.exp.groupby(carrying_name, id_exp_name, get_attachment_frame4each_group)
        res = res[res[:, -1] != -1, :]
        self.exp.add_new_dataset_from_array(array=res, name=result_name, index_names=[id_exp_name, 'th'],
                                            column_names=result_name, category=self.category,
                                            label=label, description=description)
        self.exp.write(result_name)

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

    def compute_non_outside_ant_carrying_intervals(self, redo=False, redo_hist=False):
        carrying_name = 'carrying_intervals'
        outside_ant_name = 'from_outside'
        result_name = 'non_outside_ant_carrying_intervals'

        bins = np.arange(2, 1e2, 0.5)
        hist_label = 'Histogram of carrying time intervals of non outside ants'
        hist_description = 'Histogram of the time intervals, while an ant not from outside is carrying the food'

        if redo:
            self.exp.load([carrying_name, outside_ant_name])
            self.exp.add_copy(old_name=carrying_name, new_name=result_name, category=self.category,
                              label='Carrying intervals of non outside ants',
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

    def compute_ant_attachments(self):
        carrying_name = 'carrying'
        result_name = 'ant_attachments'

        label = 'ant attachment time series'
        description = 'Time series where 1 is when an ant attaches to the food and 0 when not'

        food_name = 'food_x'
        self.exp.load([food_name, carrying_name])
        df_res = self.exp.get_df(food_name).copy().astype(int)
        df_res.columns = [result_name]
        df_res[:] = 0
        df_res[id_ant_name] = -1

        df_carr = self.exp.get_df(carrying_name).copy()
        df_carr.reset_index(inplace=True)
        df_carr[id_frame_name] += 1
        df_carr.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)

        df_carr = df_carr.reindex(self.exp.get_index(carrying_name))
        self.exp.get_data_object(carrying_name).df -= df_carr
        self.exp.get_data_object(carrying_name).df = self.exp.get_data_object(carrying_name).df.mask(df_carr.isna(), 0)

        def interval4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            frames = set(df_res.loc[id_exp, :].index.get_level_values(id_frame_name))
            print(id_exp)
            df2 = df.loc[id_exp, :].copy()
            df2 = df2.mask(df2[carrying_name] < 0, 0)
            df2 = df2.sum(level=id_frame_name)

            list_frames1 = set(df2[df2 == 1].dropna().index) & frames
            list_frames2 = set(df2[df2 == 2].dropna().index) & frames

            df3 = df.loc[id_exp, :, :][df.loc[id_exp, :, :] == 1].dropna()
            df3.reset_index(inplace=True)
            df3.drop(columns='carrying', inplace=True)
            df3.set_index(id_frame_name, inplace=True)

            for frame in list_frames1:
                id_ant = int(df3.loc[frame])
                df_res.loc[id_exp, frame] = [1, id_ant]

            for frame in list_frames2:
                id_ants = list(df3.loc[frame, id_ant_name])
                df_res.loc[id_exp, frame] = [1, id_ants[0]]
                df_res.loc[id_exp, frame+2] = [1, id_ants[1]]

            return df

        self.exp.groupby(carrying_name, id_exp_name, interval4each_group)

        df_res.reset_index(inplace=True)
        df_res.set_index([id_exp_name, id_frame_name, id_ant_name], inplace=True)
        self.exp.add_new_dataset_from_df(df_res, name=result_name, category=self.category,
                                         label=label, description=description)
        self.exp.write(result_name)
        self.exp.remove_object(carrying_name)

    def compute_outside_ant_attachments(self):
        carrying_name = 'carrying'
        from_outside_name = 'from_outside'
        result_name = 'outside_ant_attachments'

        label = 'outside ant attachment time series'
        description = 'Time series where 1 is when an ant coming from outside attaches to the food and 0 when not'

        food_name = 'food_x'
        self.exp.load([food_name, carrying_name, from_outside_name])
        df_res = self.exp.get_df(food_name).copy().astype(int)
        df_res.columns = [result_name]
        df_res[:] = 0
        df_res[id_ant_name] = -1

        df_carr = self.exp.get_df(carrying_name).copy()
        df_carr.reset_index(inplace=True)
        df_carr[id_frame_name] += 1
        df_carr.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)

        df_carr = df_carr.reindex(self.exp.get_index(carrying_name))
        self.exp.get_data_object(carrying_name).df -= df_carr
        self.exp.get_data_object(carrying_name).df = self.exp.get_data_object(carrying_name).df.mask(df_carr.isna(), 0)

        def interval4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            frames = set(df_res.loc[id_exp, :].index.get_level_values(id_frame_name))
            print(id_exp)
            df2 = df.loc[id_exp, :].copy()
            df2 = df2.mask(df2[carrying_name] < 0, 0)
            df2 = df2.sum(level=id_frame_name)

            list_frames1 = set(df2[df2 == 1].dropna().index) & frames
            list_frames2 = set(df2[df2 == 2].dropna().index) & frames

            df3 = df.loc[id_exp, :, :][df.loc[id_exp, :, :] == 1].dropna()
            df3.reset_index(inplace=True)
            df3.drop(columns='carrying', inplace=True)
            df3.set_index(id_frame_name, inplace=True)

            for frame in list_frames1:
                id_ant = int(df3.loc[frame])
                from_outside = self.exp.get_value(from_outside_name, (id_exp, id_ant))
                if from_outside == 1:
                    df_res.loc[id_exp, frame] = [1, id_ant]

            for frame in list_frames2:
                id_ants = list(df3.loc[frame, id_ant_name])

                from_outside = self.exp.get_value(from_outside_name, (id_exp, id_ants[0]))
                if from_outside == 1:
                    df_res.loc[id_exp, frame] = [1, id_ants[0]]

                from_outside = self.exp.get_value(from_outside_name, (id_exp, id_ants[1]))
                if from_outside == 1:
                    df_res.loc[id_exp, frame+2] = [1, id_ants[1]]

            return df

        self.exp.groupby(carrying_name, id_exp_name, interval4each_group)

        df_res.reset_index(inplace=True)
        df_res.set_index([id_exp_name, id_frame_name, id_ant_name], inplace=True)
        self.exp.add_new_dataset_from_df(df_res, name=result_name, category=self.category,
                                         label=label, description=description)
        self.exp.write(result_name)
        self.exp.remove_object(carrying_name)

    def compute_non_outside_ant_attachments(self):
        carrying_name = 'carrying'
        from_outside_name = 'from_outside'
        result_name = 'non_outside_ant_attachments'

        label = 'inside ant attachment time series'
        description = 'Time series where 1 is when an ant coming from inside attaches to the food and 0 when not'

        food_name = 'food_x'
        self.exp.load([food_name, carrying_name, from_outside_name])
        df_res = self.exp.get_df(food_name).copy().astype(int)
        df_res.columns = [result_name]
        df_res[:] = 0
        df_res[id_ant_name] = -1

        df_carr = self.exp.get_df(carrying_name).copy()
        df_carr.reset_index(inplace=True)
        df_carr[id_frame_name] += 1
        df_carr.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)

        df_carr = df_carr.reindex(self.exp.get_index(carrying_name))
        self.exp.get_data_object(carrying_name).df -= df_carr
        self.exp.get_data_object(carrying_name).df = self.exp.get_data_object(carrying_name).df.mask(df_carr.isna(), 0)

        def interval4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            frames = set(df_res.loc[id_exp, :].index.get_level_values(id_frame_name))
            print(id_exp)
            df2 = df.loc[id_exp, :].copy()
            df2 = df2.mask(df2[carrying_name] < 0, 0)
            df2 = df2.sum(level=id_frame_name)

            list_frames1 = set(df2[df2 == 1].dropna().index) & frames
            list_frames2 = set(df2[df2 == 2].dropna().index) & frames

            df3 = df.loc[id_exp, :, :][df.loc[id_exp, :, :] == 1].dropna()
            df3.reset_index(inplace=True)
            df3.drop(columns='carrying', inplace=True)
            df3.set_index(id_frame_name, inplace=True)

            for frame in list_frames1:
                id_ant = int(df3.loc[frame])
                from_outside = self.exp.get_value(from_outside_name, (id_exp, id_ant))
                if from_outside == 0:
                    df_res.loc[id_exp, frame] = [1, id_ant]

            for frame in list_frames2:
                id_ants = list(df3.loc[frame, id_ant_name])

                from_outside = self.exp.get_value(from_outside_name, (id_exp, id_ants[0]))
                if from_outside == 0:
                    df_res.loc[id_exp, frame] = [1, id_ants[0]]

                from_outside = self.exp.get_value(from_outside_name, (id_exp, id_ants[1]))
                if from_outside == 0:
                    df_res.loc[id_exp, frame + 2] = [1, id_ants[1]]

            return df

        self.exp.groupby(carrying_name, id_exp_name, interval4each_group)

        df_res.reset_index(inplace=True)
        df_res.set_index([id_exp_name, id_frame_name, id_ant_name], inplace=True)
        self.exp.add_new_dataset_from_df(df_res, name=result_name, category=self.category,
                                         label=label, description=description)
        self.exp.write(result_name)
        self.exp.remove_object(carrying_name)

    def compute_attachment_intervals(self, redo=False, redo_hist=False):

        result_name = 'ant_attachment_intervals'
        name = 'ant_attachments'

        label = 'Between attachment intervals'
        description = 'Time intervals between attachment intervals (s)'

        self.__compute_attachment_intervals(description, label, name, redo, redo_hist, result_name)

    def compute_outside_attachment_intervals(self, redo=False, redo_hist=False):

        result_name = 'outside_ant_attachment_intervals'
        name = 'outside_ant_attachments'

        label = 'Between outside attachment intervals'
        description = 'Time intervals between outside attachment intervals (s)'

        self.__compute_attachment_intervals(description, label, name, redo, redo_hist, result_name)

    def compute_non_outside_attachment_intervals(self, redo=False, redo_hist=False):

        result_name = 'non_outside_ant_attachment_intervals'
        name = 'non_outside_ant_attachments'

        label = 'Between non outside attachment intervals'
        description = 'Time non intervals between outside attachment intervals (s)'

        self.__compute_attachment_intervals(description, label, name, redo, redo_hist, result_name)

    def __compute_attachment_intervals(self, description, label, name, redo, redo_hist, result_name):
        bins = np.arange(0, 20, 0.5)
        if redo is True:
            self.exp.load(name)
            df = 1-self.exp.get_df(name)
            df.index = df.index.droplevel('id_ant')
            self.exp.add_new1d_from_df(df, 'temp', 'CharacteristicTimeSeries1d', replace=True)

            temp_name = self.exp.compute_time_intervals(name_to_intervals='temp', replace=True)

            df_attach = self.exp.get_df(name).copy()
            df_attach.loc[:, :, -1] = np.nan
            df_attach.dropna(inplace=True)
            df_attach.reset_index(inplace=True)
            df_attach.set_index([id_exp_name, id_frame_name], inplace=True)
            df_attach.drop(columns=name, inplace=True)
            df_attach = df_attach.reindex(self.exp.get_index(temp_name))

            df_res = self.exp.get_df(temp_name).copy()
            df_res[id_ant_name] = df_attach[id_ant_name]
            df_res.reset_index(inplace=True)
            df_res.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)
            df_res.sort_index(inplace=True)

            self.exp.add_new1d_from_df(df=df_res, name=result_name, object_type='Events1d', category=self.category,
                                       label=label, description=description)

            self.exp.write(result_name)
            self.exp.remove_object(name)

        else:
            self.exp.load(result_name)
        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(yscale='log', xlabel='Intervals (s)', ylabel='PDF', title='', label='PDF')
        plotter.plot_fit(typ='exp', preplot=(fig, ax), window=[0, 18])
        plotter.save(fig)

    def compute_isolated_ant_carrying_intervals(self):
        attachment_name = 'ant_attachment_intervals'
        result_name = 'isolated_ant_carrying_intervals'
        dt1 = 2
        dt2 = 2

        label = 'Isolated ant carrying intervals'
        description = 'Carrying intervals starting at a time' \
                      ' such that no other attachments occurred '+str(dt1)+'s before and '+str(dt2)+'s after'

        self.__isolated_attachments(dt1, dt2, attachment_name, description, label, result_name)

    def compute_isolated_outside_ant_carrying_intervals(self):
        attachment_name = 'outside_ant_attachment_intervals'
        result_name = 'isolated_outside_ant_carrying_intervals'
        dt1 = 2
        dt2 = 2

        label = 'Isolated outside ant carrying intervals'
        description = 'Carrying intervals of outside ant starting at a time' \
                      ' such that no other attachments occurred '+str(dt1)+'s before and '+str(dt2)+'s after'

        self.__isolated_attachments(dt1, dt2, attachment_name, description, label, result_name)

    def compute_isolated_non_outside_ant_carrying_intervals(self):
        attachment_name = 'non_outside_ant_attachment_intervals'
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
            self.exp.hist1d_evolution(name_to_hist=name, start_index_intervals=start_frame_intervals,
                                      end_index_intervals=end_frame_intervals, bins=bins,
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
            self.exp.hist1d_evolution(name_to_hist=name, start_index_intervals=start_frame_intervals,
                                      end_index_intervals=end_frame_intervals, bins=bins,
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

            try:
                ax.plot(self.exp.get_df(res_vel_name).loc[(vel, error), '0'],
                        self.exp.get_df(res_error_name).loc[(vel, error), '0'], 'o', c=c)
            except KeyError:
                ax.plot(self.exp.get_df(res_vel_name).loc[(vel, error), 0],
                        self.exp.get_df(res_error_name).loc[(vel, error), 0], 'o', c=c)

        ax.set_xlim((0, 8))
        ax.set_ylim((0, np.pi))
        ax.set_xticks(vel_intervals)
        ax.set_yticks(error_intervals)
        ax.grid()
        plotter.save(fig, name='mean_food_velocity_vector_length_vs_food_direction_error_around_outside_attachments')

    def __reindexing_food_xy(self, name):
        id_exps = self.exp.get_df(name).index.get_level_values(id_exp_name)
        id_ants = self.exp.get_df(name).index.get_level_values(id_ant_name)
        frames = self.exp.get_df(name).index.get_level_values(id_frame_name)
        idxs = pd.MultiIndex.from_tuples(list(zip(id_exps, frames)), names=[id_exp_name, id_frame_name])

        df_d = self.exp.get_df('food_xy').copy()
        df_d = df_d.reindex(idxs)
        df_d[id_ant_name] = id_ants
        df_d.reset_index(inplace=True)
        df_d.columns = [id_exp_name, id_frame_name, 'x', 'y', id_ant_name]
        df_d.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)
        return df_d

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
                                label=label, description=description)
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

    def compute_nb_attachments_evol(self, redo=False):

        name = 'ant_attachments'
        result_name = 'nb_attachments_evol'
        init_frame_name = 'food_first_frame'

        dx = 0.01
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(0, 4., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of attachments in a 10s period over time'
        description = 'Number of ants attaching to the food in a 10s period over time'

        if redo:
            self._mean_evol_for_nb_attachment_in_10s(name, result_name, init_frame_name, start_frame_intervals,
                                                     end_frame_intervals, label, description)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Number of attachments',
                               label_suffix='s', marker='')
        plotter.plot_smooth(preplot=(fig, ax), window=50, c='orange', label='mean smoothed')
        plotter.save(fig)

    def compute_nb_outside_attachments_evol(self, redo=False):

        name = 'outside_ant_attachments'
        result_name = 'nb_outside_attachments_evol'
        init_frame_name = 'food_first_frame'

        dx = 0.01
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(0, 4., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of outside attachments in a 10s period over time'
        description = 'Number of outside ants attaching to the food in a 10s period over time'

        if redo:
            self._mean_evol_for_nb_attachment_in_10s(name, result_name, init_frame_name, start_frame_intervals,
                                                     end_frame_intervals, label, description)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Number of attachments',
                               label_suffix='s', marker='')
        plotter.save(fig)

    def _mean_evol_for_nb_attachment_in_10s(self, name, result_name, init_frame_name, start_frame_intervals,
                                            end_frame_intervals, label, description):

        self.exp.load(name)

        self.change_first_frame(name, init_frame_name)

        x = (end_frame_intervals + start_frame_intervals) / 2. / 100.
        y = np.zeros(len(start_frame_intervals))
        for i in range(len(start_frame_intervals)):
            frame0 = int(start_frame_intervals[i])
            frame1 = int(end_frame_intervals[i])

            df = self.exp.get_df(name).loc[pd.IndexSlice[:, frame0:frame1], :]
            n_exp = len(set(df.index.get_level_values(id_exp_name)))
            y[i] = np.sum(df) / float(n_exp)
        df = pd.DataFrame(y, index=x)
        self.exp.add_new_dataset_from_df(df=df, name=result_name, category=self.category,
                                         label=label, description=description)
        self.exp.write(result_name)
        self.exp.remove_object(name)

    def compute_nb_outside_attachments_evol_around_first_attachment(self, redo=False):

        name = 'outside_ant_attachments'
        result_name = 'nb_outside_attachments_evol_around_first_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.01
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(0.01, 4., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of outside attachments in a 10s period over time'
        description = 'Number of outside ants attaching to the food in a 10s period over time'

        if redo:
            self._mean_evol_for_nb_attachment_in_10s(name, result_name, init_frame_name, start_frame_intervals,
                                                     end_frame_intervals, label, description)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Mean of the number of attachments', label='Mean',
                               label_suffix='s', marker='', title='')
        # plotter.plot_smooth(preplot=(fig, ax), window=50, c='orange', label='mean smoothed')
        plotter.plot_fit(preplot=(fig, ax), typ='log', window=[0, 550], label='log fit')
        plotter.draw_legend(ax)
        ax.set_ylim((0, 6))
        plotter.save(fig)

    def compute_nb_non_outside_attachments_evol_around_first_attachment(self, redo=False):

        name = 'non_outside_ant_attachments'
        result_name = 'nb_non_outside_attachments_evol_around_first_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.01
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(-2, 4., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of non outside attachments in a 10s period over time'
        description = 'Number of non outside ants attaching to the food in a 10s period over time'

        if redo:
            self._mean_evol_for_nb_attachment_in_10s(name, result_name, init_frame_name, start_frame_intervals,
                                                     end_frame_intervals, label, description)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Mean of the number of attachments', label='Mean',
                               label_suffix='s', marker='')
        plotter.plot_smooth(preplot=(fig, ax), window=50, c='orange', label='mean smoothed')
        plotter.draw_vertical_line(ax, label='first outside attachment')
        ax.legend()
        plotter.save(fig)

    def compute_ratio_outside_attachments_evol_around_first_attachment(self, redo=False):

        result_name = 'ratio_outside_attachments_evol_around_first_attachment'
        label = 'Ratio of outside attachments in a 10s period over time'
        description = 'Ratio of number of outside attachments on total number of attachments in a 10s period over time'

        if redo:
            name = 'nb_attachments_evol_around_first_attachment'
            outside_name = 'nb_outside_attachments_evol_around_first_attachment'
            self.exp.load([name, outside_name])

            self.exp.add_copy(old_name=outside_name, new_name=result_name, category=self.category,
                              label=label, description=description)

            self.exp.operation_between_2names(result_name, name, lambda a, b: a/b)
            self.exp.write(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Mean of the number of attachments', label='Mean',
                               label_suffix='s', marker='')
        plotter.plot_smooth(preplot=(fig, ax), window=50, c='orange', label='mean smoothed')
        plotter.draw_vertical_line(ax, label='first outside attachment')
        # plotter.plot_fit(preplot=(fig, ax), typ='log', window=[220, 550], label='log fit')
        ax.legend()
        plotter.save(fig)

    def compute_nb_attachments_evol_around_first_attachment(self, redo=False):

        name = 'ant_attachments'
        result_name = 'nb_attachments_evol_around_first_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.01
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(-2, 4., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of attachments in a 10s period over time'
        description = 'Number of ants attaching to the food in a 10s period over time'

        if redo:
            self._mean_evol_for_nb_attachment_in_10s(name, result_name, init_frame_name, start_frame_intervals,
                                                     end_frame_intervals, label, description)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Mean of the number of attachments', label='Mean',
                               label_suffix='s', marker='', title='')
        plotter.draw_vertical_line(ax, label='First outside ant attachment')
        ax.set_ylim((0, 7))
        plotter.save(fig)
