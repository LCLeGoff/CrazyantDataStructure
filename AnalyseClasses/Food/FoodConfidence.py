import numpy as np
import pandas as pd

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_frame_name
from Tools.MiscellaneousTools.Geometry import distance_df
from Tools.Plotter.Plotter import Plotter


class AnalyseFoodConfidence(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'FoodConfidence'

    def compute_w1s_food_crossed_distance(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 1
        self.__compute_crossed_distance(w, redo, redo_plot_indiv, redo_hist)

    def compute_w2s_food_crossed_distance(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 2
        self.__compute_crossed_distance(w, redo, redo_plot_indiv, redo_hist)

    def compute_w10s_food_crossed_distance(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 10
        self.__compute_crossed_distance(w, redo, redo_plot_indiv, redo_hist)

    def compute_w16s_food_crossed_distance(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 16
        self.__compute_crossed_distance(w, redo, redo_plot_indiv, redo_hist)

    def compute_w30s_food_crossed_distance(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 30
        self.__compute_crossed_distance(w, redo, redo_plot_indiv, redo_hist)

    def __compute_crossed_distance(self, w, redo, redo_plot_indiv, redo_hist):
        result_name = 'w' + str(w) + 's_food_crossed_distance'
        hist_name = result_name+'_hist'

        bins = 'fd'
        label = 'Food crossed distance (mm/s)'
        description = 'Distance crossed by te food during ' + str(w) + ' s (mm/s)'
        if redo:
            name_x = 'mm10_food_x'
            name_y = 'mm10_food_y'

            name_xy = 'food_xy'
            self.exp.load_as_2d(name1=name_x, name2=name_y, result_name=name_xy, replace=True)
            self.exp.load('fps')

            self.exp.add_copy(old_name=name_x, new_name=result_name, category=self.category,
                              label=label, description=description)
            self.exp.get_data_object(result_name).df[:] = np.nan

            w2 = int(w / 2)

            def compute_crossed_distance4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                frame0 = df.index.get_level_values(id_frame_name)[0]
                frame1 = df.index.get_level_values(id_frame_name)[-1]
                print(id_exp)

                fps = self.exp.get_value('fps', id_exp)
                w_in_f = int(w * fps)
                w_in_f2 = int(w_in_f / 2)

                df2 = df.loc[id_exp, :]

                xy1 = df2.copy()
                xy2 = df2.copy()
                xy1.index += w_in_f2
                xy2.index -= w_in_f2

                norm = df2.copy()
                norm = norm[name_x]
                norm[:] = w

                frame = frame0 + w_in_f2 - 1
                temp_df = (df2.loc[:frame]).copy()
                xy1 = pd.concat([temp_df, xy1])
                xy1.loc[:frame, name_x] = xy1.loc[frame0, name_x]
                xy1.loc[:frame, name_y] = xy1.loc[frame0, name_y]
                norm.loc[:frame] = (np.array(norm.loc[:frame].index)-frame0) / fps + w2

                frame = frame1 - w_in_f2 + 1
                temp_df = (df2.loc[frame:]).copy()
                xy2 = pd.concat([xy2, temp_df])
                xy2.loc[frame:, name_x] = xy2.loc[frame1, name_x]
                xy2.loc[frame:, name_y] = xy2.loc[frame1, name_y]
                norm.loc[frame:] = (frame1-np.array(norm.loc[frame:].index)) / fps + w2

                dist = distance_df(xy1, xy2)/norm
                dist = dist.reindex(df.index.get_level_values(id_frame_name))
                self.exp.get_df(result_name).loc[id_exp, :] = np.array(np.around(dist, 5))

            self.exp.groupby(name_xy, id_exp_name, compute_crossed_distance4each_group)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        ylabel = 'Crossed Distance'
        self.__plot_indiv(result_name, ylabel, redo, redo_plot_indiv)

        self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter = Plotter(self.exp.root, self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot()
        plotter.save(fig)

    def compute_w1s_food_total_crossed_distance(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 1
        self.__compute_total_crossed_distance(w, redo, redo_plot_indiv, redo_hist)

    def compute_w2s_food_total_crossed_distance(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 2
        self.__compute_total_crossed_distance(w, redo, redo_plot_indiv, redo_hist)

    def compute_w10s_food_total_crossed_distance(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 10
        self.__compute_total_crossed_distance(w, redo, redo_plot_indiv, redo_hist)

    def compute_w16s_food_total_crossed_distance(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 16
        self.__compute_total_crossed_distance(w, redo, redo_plot_indiv, redo_hist)

    def compute_w30s_food_total_crossed_distance(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 30
        self.__compute_total_crossed_distance(w, redo, redo_plot_indiv, redo_hist)

    def __compute_total_crossed_distance(self, w, redo, redo_plot_indiv, redo_hist):

        result_name = 'w' + str(w) + 's_food_total_crossed_distance'
        hist_name = result_name + '_hist'
        bins = np.arange(40)
        label = 'Food total crossed distance (mm/s)'
        description = 'Distance total crossed by te food during ' + str(w) + ' s (mm/s)'

        if redo:
            name_x = 'mm10_food_x'
            name_y = 'mm10_food_y'

            name_xy = 'food_xy'
            self.exp.load_as_2d(name1=name_x, name2=name_y, result_name=name_xy, replace=True)
            self.exp.load('fps')

            self.exp.add_copy(old_name=name_x, new_name=result_name, category=self.category,
                              label=label, description=description)
            self.exp.get_data_object(result_name).df[:] = np.nan

            def compute_crossed_distance4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                frame0 = df.index.get_level_values(id_frame_name)[0]
                frame1 = df.index.get_level_values(id_frame_name)[-1]
                print(id_exp)

                fps = self.exp.get_value('fps', id_exp)
                w_in_f = int(w * fps)

                df2 = df.loc[id_exp, :]

                xy1 = df2.loc[frame0 + 1:]
                xy2 = df2.loc[:frame1-1]
                xy1.index -= 1
                xy2.index += 1
                dist = distance_df(xy1, xy2)*fps/2.
                dist = dist.rolling(window=w_in_f, center=True).apply(np.nanmean)
                dist = dist.reindex(df.index.get_level_values(id_frame_name))
                self.exp.get_df(result_name).loc[id_exp, :] = np.array(np.around(dist, 5))

            self.exp.groupby(name_xy, id_exp_name, compute_crossed_distance4each_group)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        ylabel = 'Total crossed Distance'
        self.__plot_indiv(result_name, ylabel, redo, redo_plot_indiv)

        self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter = Plotter(self.exp.root, self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot()
        plotter.save(fig)

    def __plot_indiv(self, result_name, ylabel, redo, redo_plot_indiv):
        if redo or redo_plot_indiv:
            attachment_name = 'outside_carrying_intervals'
            self.exp.load(['fps', attachment_name])

            def plot_indiv(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                fps = self.exp.get_value('fps', id_exp)

                df2 = df.loc[id_exp, :]
                df2.index /= fps
                self.exp.add_new_dataset_from_df(df=df2, name=str(id_exp),
                                                 category=self.category, replace=True)

                plotter2 = Plotter(self.exp.root, self.exp.get_data_object(str(id_exp)))
                fig2, ax2 = plotter2.plot(xlabel='Time (s)', ylabel=ylabel, marker='',
                                          title_prefix='Exp ' + str(id_exp) + ': ')

                attachments = self.exp.get_df(attachment_name).loc[id_exp, :]
                attachments.reset_index(inplace=True)
                attachments = np.array(attachments) / fps

                colors = plotter2.color_object.create_cmap('hot_r', set(list(attachments[:, 0])))
                for id_ant, frame, inter in attachments:
                    ax2.axvline(frame, c=colors[str(id_ant)], alpha=0.5)

                plotter2.save(fig2, name=id_exp, sub_folder=result_name)

            self.exp.groupby(result_name, id_exp_name, plot_indiv)

    def compute_w1s_food_path_efficiency(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 1
        self.__compute_path_efficiency(w, redo, redo_plot_indiv, redo_hist)

    def compute_w2s_food_path_efficiency(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 2
        self.__compute_path_efficiency(w, redo, redo_plot_indiv, redo_hist)

    def compute_w10s_food_path_efficiency(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 10
        self.__compute_path_efficiency(w, redo, redo_plot_indiv, redo_hist)

    def compute_w16s_food_path_efficiency(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 16
        self.__compute_path_efficiency(w, redo, redo_plot_indiv, redo_hist)

    def compute_w30s_food_path_efficiency(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 30
        self.__compute_path_efficiency(w, redo, redo_plot_indiv, redo_hist)

    def __compute_path_efficiency(self, w, redo, redo_plot_indiv, redo_hist):
        dist_name = 'w' + str(w) + 's_food_crossed_distance'
        total_dist_name = 'w' + str(w) + 's_food_total_crossed_distance'
        result_name = 'w' + str(w) + 's_food_path_efficiency'

        hist_name = result_name + '_hist'
        bins = np.arange(0, 1, 0.01)
        label = 'Food path efficiency'
        description = 'Distance crossed by the food during ' + str(w) + ' s ' \
                      'divided by the total distance crossed by the food during the same time'
        if redo:
            self.exp.load([dist_name, total_dist_name])

            self.exp.add_copy(old_name=dist_name, new_name=result_name, category=self.category,
                              label=label, description=description)

            self.exp.get_data_object(result_name).df /= np.array(self.exp.get_df(total_dist_name))
            self.exp.get_data_object(result_name).df = self.exp.get_df(result_name).round(5)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        ylabel = 'Path efficiency'
        self.__plot_indiv(result_name, ylabel, redo, redo_plot_indiv)

        self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter = Plotter(self.exp.root, self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(normed=True)
        plotter.save(fig)

    def compute_w10s_food_crossed_distance_resolution1pc(self, redo=False, redo_hist=False):
        w = 10
        self.__compute_crossed_distance_resolution1pc(w, redo, redo_hist)

    def compute_w16s_food_crossed_distance_resolution1pc(self, redo=False, redo_hist=False):
        w = 16
        self.__compute_crossed_distance_resolution1pc(w, redo, redo_hist)

    def compute_w30s_food_crossed_distance_resolution1pc(self, redo=False, redo_hist=False):
        w = 30
        self.__compute_crossed_distance_resolution1pc(w, redo, redo_hist)

    def __compute_crossed_distance_resolution1pc(self, w, redo, redo_hist):
        result_name = 'w' + str(w) + 's_food_crossed_distance_resolution1pc'
        hist_name = result_name+'_hist'

        bins = 'fd'
        label = 'Food crossed distance (mm/s)'
        description = 'Distance crossed by te food during ' + str(w) + ' s (mm/s) using one data per second'
        if redo:
            name_x = 'mm10_food_x'
            name_y = 'mm10_food_y'

            name_xy = 'food_xy'
            self.exp.load_as_2d(name1=name_x, name2=name_y, result_name=name_xy, replace=True)
            self.exp.load('fps')
            self.exp.change_df(name_xy, self.exp.get_df(name_xy).iloc[::100])

            df_res = self.exp.get_df(name_x).iloc[::100].copy()
            df_res.columns = [result_name]
            df_res[:] = np.nan

            w2 = int(w / 2)

            def compute_crossed_distance4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                frame0 = df.index.get_level_values(id_frame_name)[0]
                frame1 = df.index.get_level_values(id_frame_name)[-1]
                print(id_exp)

                fps = self.exp.get_value('fps', id_exp)
                w_in_f = int(w * fps)
                w_in_f2 = int(w_in_f / 2)

                df2 = df.loc[id_exp, :]

                xy1 = df2.copy()
                xy2 = df2.copy()
                xy1.index += w_in_f2
                xy2.index -= w_in_f2

                norm = df2.copy()
                norm = norm[name_x]
                norm[:] = w

                frame = frame0 + w_in_f2 - 1
                temp_df = (df2.loc[:frame]).copy()
                xy1 = pd.concat([temp_df, xy1])
                xy1.loc[:frame, name_x] = xy1.loc[frame0, name_x]
                xy1.loc[:frame, name_y] = xy1.loc[frame0, name_y]
                norm.loc[:frame] = (np.array(norm.loc[:frame].index)-frame0) / fps + w2

                frame = frame1 - w_in_f2 + 1
                temp_df = (df2.loc[frame:]).copy()
                xy2 = pd.concat([xy2, temp_df])
                xy2.loc[frame:, name_x] = xy2.loc[frame1, name_x]
                xy2.loc[frame:, name_y] = xy2.loc[frame1, name_y]
                norm.loc[frame:] = (frame1-np.array(norm.loc[frame:].index)) / fps + w2

                dist = distance_df(xy1, xy2)/norm
                dist = dist.reindex(df.index.get_level_values(id_frame_name))
                df_res.loc[id_exp, :] = np.array(np.around(dist, 5))

                return df

            self.exp.groupby(name_xy, id_exp_name, compute_crossed_distance4each_group)

            self.exp.add_new_dataset_from_df(
                df=df_res, name=result_name, category=self.category, label=label, description=description)

            self.exp.write(result_name)
            self.exp.remove_object(name_xy)
        else:
            self.exp.load(result_name)

        self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter = Plotter(self.exp.root, self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot()
        plotter.save(fig)

    def compute_w10s_food_total_crossed_distance_resolution1pc(self, redo=False, redo_hist=False):
        w = 10
        self.__compute_total_crossed_distance_resolution1pc(w, redo, redo_hist)

    def compute_w16s_food_total_crossed_distance_resolution1pc(self, redo=False, redo_hist=False):
        w = 16
        self.__compute_total_crossed_distance_resolution1pc(w, redo, redo_hist)

    def compute_w30s_food_total_crossed_distance_resolution1pc(self, redo=False, redo_hist=False):
        w = 30
        self.__compute_total_crossed_distance_resolution1pc(w, redo, redo_hist)

    def __compute_total_crossed_distance_resolution1pc(self, w, redo, redo_hist):

        result_name = 'w' + str(w) + 's_food_total_crossed_distance_resolution1pc'
        hist_name = result_name + '_hist'
        bins = np.arange(40)
        label = 'Food total crossed distance (mm/s)'
        description = 'Distance total crossed by te food during ' + str(w) + ' s (mm/s) using one data per second'

        if redo:
            name_x = 'mm10_food_x'
            name_y = 'mm10_food_y'

            name_xy = 'food_xy'
            self.exp.load_as_2d(name1=name_x, name2=name_y, result_name=name_xy, replace=True)
            self.exp.change_df(name_xy, self.exp.get_df(name_xy).iloc[::100])

            df_res = self.exp.get_df(name_x).iloc[::100].copy()
            df_res.columns = [result_name]
            df_res[:] = np.nan

            def compute_total_crossed_distance4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                frame0 = df.index.get_level_values(id_frame_name)[0]
                frame1 = df.index.get_level_values(id_frame_name)[-1]
                print(id_exp)

                df2 = df.loc[id_exp, :]

                xy1 = df2.loc[frame0 + 50:]
                xy2 = df2.loc[:frame1-50]
                xy1.index -= 50
                xy2.index += 50
                dist = distance_df(xy1, xy2)
                dist = dist.rolling(window=w, center=True).apply(np.nanmean)
                dist.index -= 50
                dist = dist.reindex(df.index.get_level_values(id_frame_name))
                df_res.loc[id_exp, :] = np.array(np.around(dist, 5))

            self.exp.groupby(name_xy, id_exp_name, compute_total_crossed_distance4each_group)

            self.exp.add_new_dataset_from_df(
                df=df_res, name=result_name, category=self.category, label=label, description=description)

            self.exp.write(result_name)
            self.exp.remove_object(name_xy)
        else:
            self.exp.load(result_name)

        self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter = Plotter(self.exp.root, self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(normed=True)
        plotter.save(fig)

    def compute_w10s_food_path_efficiency_resolution1pc(self, redo=False, redo_hist=False):
        w = 10
        self.__compute_path_efficiency_resolution1pc(w, redo, redo_hist)

    def compute_w16s_food_path_efficiency_resolution1pc(self, redo=False, redo_hist=False):
        w = 16
        self.__compute_path_efficiency_resolution1pc(w, redo, redo_hist)

    def compute_w30s_food_path_efficiency_resolution1pc(self, redo=False, redo_hist=False):
        w = 30
        self.__compute_path_efficiency_resolution1pc(w, redo, redo_hist)

    def __compute_path_efficiency_resolution1pc(self, w, redo, redo_hist):
        dist_name = 'w' + str(w) + 's_food_crossed_distance_resolution1pc'
        total_dist_name = 'w' + str(w) + 's_food_total_crossed_distance_resolution1pc'
        result_name = 'w' + str(w) + 's_food_path_efficiency_resolution1pc'

        hist_name = result_name + '_hist'
        bins = np.arange(0, 1, 0.01)
        label = 'Food path efficiency'
        description = 'Distance crossed by the food during ' + str(w) + ' s ' \
                      'divided by the total distance crossed by the food during the same time' \
                      'using one data per second'
        if redo:
            self.exp.load([dist_name, total_dist_name])

            self.exp.add_copy(old_name=dist_name, new_name=result_name, category=self.category,
                              label=label, description=description)

            self.exp.get_data_object(result_name).df /= np.array(self.exp.get_df(total_dist_name))
            self.exp.get_data_object(result_name).df = self.exp.get_df(result_name).round(5)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter = Plotter(self.exp.root, self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(normed=True)
        plotter.save(fig)

    def compute_w1s_food_path_efficiency_around_attachments(self):
        variable_name = 'w1s_food_path_efficiency'
        attachment_name = '%s_attachment_intervals'
        result_name = variable_name + '_around_%s_attachments'

        result_label = 'Food path efficiency around %s attachments'
        result_description = 'Food path efficiency with window 1s' \
                             ' before and after an ant coming from %s ant attached to the food'
        for suff in ['outside', 'inside']:

            self.__gather_exp_frame_indexed_around_attachments(
                variable_name, attachment_name % suff, result_name % suff,
                result_label % suff, result_description % suff)

    def compute_w1s_food_path_efficiency_around_leading_attachments(self):
        variable_name = 'w1s_food_path_efficiency'
        attachment_name = '%s_leading_attachment_intervals'
        result_name = variable_name + '_around_%s_leading_attachments'

        result_label = 'Food path efficiency around %s leading attachments'
        result_description = 'Food path efficiency with window 1s' \
                             ' before and after an ant coming from %s ant attached to the food and has' \
                             ' an influence on it'
        for suff in ['outside', 'inside']:
            self.__gather_exp_frame_indexed_around_attachments(
                variable_name, attachment_name % suff, result_name % suff,
                result_label % suff, result_description % suff)

    def compute_w2s_food_path_efficiency_around_leading_attachments(self):
        variable_name = 'w2s_food_path_efficiency'
        attachment_name = '%s_leading_attachment_intervals'
        result_name = variable_name + '_around_%s_leading_attachments'

        result_label = 'Food path efficiency around %s leading attachments'
        result_description = 'Food path efficiency with window 2s' \
                             ' before and after an ant coming from %s ant attached to the food and has' \
                             ' an influence on it'
        for suff in ['outside', 'inside']:
            self.__gather_exp_frame_indexed_around_attachments(
                variable_name, attachment_name % suff, result_name % suff,
                result_label % suff, result_description % suff)

    def compute_w10s_food_path_efficiency_around_leading_attachments(self):
        variable_name = 'w10s_food_path_efficiency'
        attachment_name = '%s_leading_attachment_intervals'
        result_name = variable_name + '_around_%s_leading_attachments'

        result_label = 'Food path efficiency around %s leading attachments'
        result_description = 'Food path efficiency with window 10s' \
                             ' before and after an ant coming from %s ant attached to the food and has' \
                             ' an influence on it'
        for suff in ['outside', 'inside']:
            self.__gather_exp_frame_indexed_around_attachments(
                variable_name, attachment_name % suff, result_name % suff,
                result_label % suff, result_description % suff)

    def compute_w10s_food_path_efficiency_around_attachments(self):
        variable_name = 'w10s_food_path_efficiency'
        attachment_name = '%s_attachment_intervals'
        result_name = variable_name + '_around_%s_attachments'

        result_label = 'Food path efficiency around %s attachments'
        result_description = 'Food path efficiency with window 10s' \
                             ' before and after an ant coming from %s ant attached to the food'
        for suff in ['outside', 'inside']:
            self.__gather_exp_frame_indexed_around_attachments(
                variable_name, attachment_name % suff, result_name % suff,
                result_label % suff, result_description % suff)

    def __gather_exp_frame_indexed_around_attachments(
            self, variable_name, attachment_name, result_name, result_label, result_description,
            time_min=None, time_max=None):

        if time_min is None:
            time_min = -np.inf

        if time_max is None:
            time_max = np.inf

        self.exp.load(attachment_name)
        last_frame_name = 'food_exit_frames'
        first_frame_name = 'first_attachment_time_of_outside_ant'
        self.exp.load([variable_name, first_frame_name, last_frame_name, 'fps'], reload=False)

        t0, t1, dt = -10, 10, 0.1
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)
        time_intervals2 = np.array(time_intervals, dtype=str)
        time_intervals2[time_intervals2 == '-0.0'] = '0.0'

        index_values = self.exp.get_df(attachment_name).reset_index()
        index_values = index_values.set_index([id_exp_name, id_frame_name])
        index_values = index_values.sort_index()
        index_values = index_values.index
        self.exp.add_new_empty_dataset(name=result_name, index_names=[id_exp_name, id_frame_name],
                                       column_names=time_intervals2, index_values=index_values,
                                       category=self.category, label=result_label, description=result_description)

        def get_variable4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            fps = self.exp.get_value('fps', id_exp)
            first_frame = self.exp.get_value(first_frame_name, id_exp)
            last_frame = min(time_max+first_frame, self.exp.get_value(last_frame_name, id_exp))
            first_frame = max(time_min+first_frame, first_frame)

            if id_exp in self.exp.get_index(attachment_name).get_level_values(id_exp_name):
                attachment_frames = self.exp.get_df(attachment_name).loc[id_exp, :, :]
                attachment_frames = list(set(attachment_frames.index.get_level_values(id_frame_name)))
                attachment_frames.sort()

                for attach_frame in attachment_frames:

                    if last_frame > attach_frame > first_frame:
                        f0 = int(attach_frame + time_intervals[0] * fps)
                        f1 = int(attach_frame + time_intervals[-1] * fps)

                        var_df = df.loc[pd.IndexSlice[id_exp, f0:f1], :]
                        var_df = var_df.loc[id_exp, :]
                        var_df.index -= attach_frame
                        var_df.index /= fps

                        var_df = var_df.reindex(time_intervals)

                        self.exp.get_df(result_name).loc[(id_exp, attach_frame), :] = \
                            np.array(var_df[variable_name])

        self.exp.groupby(variable_name, id_exp_name, func=get_variable4each_group)
        self.exp.change_df(result_name, self.exp.get_df(result_name).dropna(how='all'))
        print(len(self.exp.get_df(result_name)))
        self.exp.write(result_name)

    def __gather_exp_frame_indexed_around_attachments_with_discrimination(
            self, variable_name, discrim_name, attachment_name, result_name, result_label, result_description,
            time_min=None, time_max=None):

        if time_min is None:
            time_min = -np.inf

        if time_max is None:
            time_max = np.inf

        last_frame_name = 'food_exit_frames'
        first_frame_name = 'first_attachment_time_of_outside_ant'
        self.exp.load([variable_name, first_frame_name, last_frame_name, 'fps', attachment_name, discrim_name])

        t0, t1, dt = -10, 10, 0.1
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)
        time_intervals2 = np.array(time_intervals, dtype=str)
        time_intervals2[time_intervals2 == '-0.0'] = '0.0'

        index_values = self.exp.get_df(attachment_name).reset_index()
        index_values = index_values.set_index([id_exp_name, id_frame_name])
        index_values = index_values.sort_index()
        index_values = index_values.index

        index_exp = index_values.get_level_values(id_exp_name)
        index_frame = index_values.get_level_values(id_frame_name)
        index_discrim = np.around(self.exp.get_df(discrim_name).loc[index_values].values.ravel(), 3)
        index_values = np.array(list(zip(index_exp, index_frame, index_discrim)))

        self.exp.add_new_empty_dataset(name=result_name, index_names=[id_exp_name, id_frame_name, discrim_name],
                                       column_names=time_intervals2, index_values=index_values,
                                       category=self.category, label=result_label, description=result_description)

        def get_variable4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            fps = self.exp.get_value('fps', id_exp)
            first_frame = self.exp.get_value(first_frame_name, id_exp)
            last_frame = min(time_max+first_frame, self.exp.get_value(last_frame_name, id_exp))
            first_frame = max(time_min+first_frame, first_frame)

            if id_exp in self.exp.get_index(attachment_name).get_level_values(id_exp_name):
                attachment_frames = self.exp.get_df(attachment_name).loc[id_exp, :, :]
                attachment_frames = list(set(attachment_frames.index.get_level_values(id_frame_name)))
                attachment_frames.sort()

                for attach_frame in attachment_frames:

                    if last_frame > attach_frame > first_frame:
                        f0 = int(attach_frame + time_intervals[0] * fps)
                        f1 = int(attach_frame + time_intervals[-1] * fps)

                        var_df = df.loc[pd.IndexSlice[id_exp, f0:f1], :]
                        var_df = var_df.loc[id_exp, :]
                        var_df.index -= attach_frame
                        var_df.index /= fps

                        var_df = var_df.reindex(time_intervals)

                        self.exp.get_df(result_name).loc[pd.IndexSlice[id_exp, attach_frame, :], :] = \
                            np.array(var_df[variable_name])

        self.exp.groupby(variable_name, id_exp_name, func=get_variable4each_group)
        self.exp.change_df(result_name, self.exp.get_df(result_name).dropna(how='all'))
        print(len(self.exp.get_df(result_name)))
        self.exp.write(result_name)

    def __compute_mean(self, time_intervals, variable_name, result_name):
        for t in time_intervals:
            tab = self.exp.get_df(variable_name)[str(t)]
            m = np.mean(tab)
            self.exp.get_df(result_name).loc[t, 'mean'] = np.around(m, 3)
            self.exp.get_df(result_name).loc[t, 'CI95_1'] = np.around(m-np.percentile(tab, 2.5), 3)
            self.exp.get_df(result_name).loc[t, 'CI95_2'] = np.around(np.percentile(tab, 97.5)-m, 3)

    def __compute_mean2(self, time_intervals, eff_intervals, variable_name, result_name):
        for j in range(len(eff_intervals)-1):
            r = 0
            eff0 = eff_intervals[j]
            eff1 = eff_intervals[j+1]
            for t in time_intervals[:-1]:
                tab = self.exp.get_df(variable_name).loc[pd.IndexSlice[:, :, eff0:eff1], str(t)]
                r += len(tab)
                m = np.mean(tab)
                self.exp.get_df(result_name).loc[eff0, t]['mean'] = np.around(m, 3)
                self.exp.get_df(result_name).loc[eff0, t]['CI95_1'] = np.around(m-np.percentile(tab, 5), 3)
                self.exp.get_df(result_name).loc[eff0, t]['CI95_2'] = np.around(np.percentile(tab, 95)-m, 3)
                # std = np.std(tab)
                # self.exp.get_df(result_name).loc[eff0, t]['CI95_1'] = np.around(std*2, 3)
                # self.exp.get_df(result_name).loc[eff0, t]['CI95_2'] = np.around(std*2, 3)
            print(eff0, r/(len(eff_intervals)-1))

    def __compute_mean_around_attachments(
            self, fct, variable_name, mean_result_name, hist_result_name,
            mean_label, mean_description, hist_label, hist_description, redo, list_exps=None):

        t0, t1, dt = -2, 8, 2.
        time_intervals = np.around(np.arange(t0, t1 + dt+1, dt), 1)

        bins = np.arange(0, 1.1, 0.1)
        hists_index_values = np.around((bins[1:] + bins[:-1]) / 2., 3)
        if redo:
            self.exp.load(variable_name)

            self.exp.add_new_empty_dataset(name=hist_result_name, index_names='food_path_efficiency',
                                           column_names=time_intervals[:-1], index_values=hists_index_values,
                                           category=self.category, label=hist_label, description=hist_description,
                                           replace=True)

            for i in range(len(time_intervals)-1):
                t0 = time_intervals[i]
                t1 = time_intervals[i+1]
                values = self.exp.get_df(variable_name).loc[pd.IndexSlice[:, :], str(t0):str(t1)].dropna()
                if list_exps is not None:
                    values = values.loc[list_exps, :]
                values = values.values.ravel()
                hist = np.histogram(values, bins=bins, normed=False)[0]

                self.exp.get_df(hist_result_name)[t0] = hist

            t0, t1, dt = -10., 10., .5
            time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)

            self.exp.add_new_empty_dataset(name=mean_result_name, index_names='time',
                                           column_names=['mean', 'CI95_1', 'CI95_2'],
                                           index_values=time_intervals, category=self.category,
                                           label=mean_label, description=mean_description, replace=True)

            fct(time_intervals, variable_name, mean_result_name)

            self.exp.remove_object(variable_name)
            self.exp.write(hist_result_name)
            self.exp.write(mean_result_name)
        else:
            self.exp.load(mean_result_name)
            self.exp.load(hist_result_name)

    def compute_mean_w1s_food_path_efficiency_around_leading_attachments(self, redo):

        variable_name = 'w1s_food_path_efficiency_around_%s_leading_attachments'
        mean_result_name = 'mean_' + variable_name
        hist_result_name = 'histograms_' + variable_name

        mean_label = 'Mean of the food path efficiency around %s leader attachments'
        mean_description = 'Mean of the food path efficiency for each time t in time_intervals,' \
                           ' which are times around leading %s attachments'

        hist_label = 'Histograms of the food path efficiency around %s leader attachments'
        hist_description = 'Histograms of the food path efficiency for each time t in time_intervals,' \
                           ' which are times around leading %s attachments'

        for suff in ['outside', 'inside']:
            self.__compute_mean_around_attachments(
                self.__compute_mean, variable_name % suff, mean_result_name % suff, hist_result_name % suff,
                mean_label % suff, mean_description % suff,
                hist_label % suff, hist_description % suff, redo)

        suff = 'outside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(hist_result_name % suff))
        fig, ax = plotter.create_plot(figsize=(9, 5), ncols=2, left=0.05)
        plotter.plot(preplot=(fig, ax[0]), title=suff, xlabel='Path Efficiency', ylabel='PDF', normed=True)
        ax[0].set_xlim(0, 1)
        suff = 'inside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(hist_result_name % suff))
        plotter.plot(preplot=(fig, ax[1]), title=suff, xlabel='Path Efficiency', ylabel='PDF', normed=True)
        ax[1].set_xlim(0, 1)
        plotter.save(fig, name=hist_result_name % '')

        suff = 'outside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(mean_result_name % suff))
        fig, ax = plotter.plot_with_error(
            xlabel='time (s)', ylabel='Mean path efficiency', title='', c='red', label=suff)

        suff = 'inside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(mean_result_name % suff))
        fig, ax = plotter.plot_with_error(
            preplot=(fig, ax), xlabel='time (s)', ylabel='Mean path efficiency', title='', c='navy', label=suff)

        ax.axvline(0, ls='--', c='k')
        ax.set_xlim((-2, 8))
        plotter.save(fig)

    def compute_mean_w1s_food_path_efficiency_around_attachments(self, redo):

        variable_name = 'w1s_food_path_efficiency_around_%s_attachments'
        mean_result_name = 'mean_' + variable_name
        hist_result_name = 'histograms_' + variable_name

        mean_label = 'Mean of the food path efficiency around %s attachments'
        mean_description = 'Mean of the food path efficiency for each time t in time_intervals,' \
                           ' which are times around %s attachments'

        hist_label = 'Histograms of the food path efficiency around %s attachments'
        hist_description = 'Histograms of the food path efficiency for each time t in time_intervals,' \
                           ' which are times around %s attachments'

        for suff in ['outside', 'inside']:
            self.__compute_mean_around_attachments(
                self.__compute_mean, variable_name % suff, mean_result_name % suff, hist_result_name % suff,
                mean_label % suff, mean_description % suff,
                hist_label % suff, hist_description % suff, redo)

        suff = 'outside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(hist_result_name % suff))
        fig, ax = plotter.create_plot(figsize=(9, 5), ncols=2, left=0.05)
        plotter.plot(preplot=(fig, ax[0]), title=suff, xlabel='Path Efficiency', ylabel='PDF', normed=True)
        ax[0].set_xlim(0, 1)
        suff = 'inside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(hist_result_name % suff))
        plotter.plot(preplot=(fig, ax[1]), title=suff, xlabel='Path Efficiency', ylabel='PDF', normed=True)
        ax[1].set_xlim(0, 1)
        plotter.save(fig, name=hist_result_name % '')

        suff = 'outside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(mean_result_name % suff))
        fig, ax = plotter.plot_with_error(
            xlabel='time (s)', ylabel='Mean path efficiency', title='', c='red', label=suff)

        suff = 'inside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(mean_result_name % suff))
        fig, ax = plotter.plot_with_error(
            preplot=(fig, ax), xlabel='time (s)', ylabel='Mean path efficiency', title='', c='navy', label=suff)

        ax.axvline(0, ls='--', c='k')
        ax.set_xlim((-2, 8))
        plotter.save(fig)

    def compute_mean_w2s_food_path_efficiency_around_leading_attachments(self, redo):

        variable_name = 'w2s_food_path_efficiency_around_%s_leading_attachments'
        mean_result_name = 'mean_' + variable_name
        hist_result_name = 'histograms_' + variable_name

        mean_label = 'Mean of the food path efficiency around %s leader attachments'
        mean_description = 'Mean of the food path efficiency for each time t in time_intervals,' \
                           ' which are times around leading %s attachments'

        hist_label = 'Histograms of the food path efficiency around %s leader attachments'
        hist_description = 'Histograms of the food path efficiency for each time t in time_intervals,' \
                           ' which are times around leading %s attachments'

        for suff in ['outside', 'inside']:
            self.__compute_mean_around_attachments(
                self.__compute_mean, variable_name % suff, mean_result_name % suff, hist_result_name % suff,
                mean_label % suff, mean_description % suff,
                hist_label % suff, hist_description % suff, redo)

        suff = 'outside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(hist_result_name % suff))
        fig, ax = plotter.create_plot(figsize=(9, 5), ncols=2, left=0.05)
        plotter.plot(preplot=(fig, ax[0]), title=suff, xlabel='Path Efficiency', ylabel='PDF', normed=True)
        ax[0].set_xlim(0, 1)
        suff = 'inside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(hist_result_name % suff))
        plotter.plot(preplot=(fig, ax[1]), title=suff, xlabel='Path Efficiency', ylabel='PDF', normed=True)
        ax[1].set_xlim(0, 1)
        plotter.save(fig, name=hist_result_name % '')

        suff = 'outside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(mean_result_name % suff))
        fig, ax = plotter.plot_with_error(
            xlabel='time (s)', ylabel='Mean path efficiency', title='', c='red', label=suff)

        suff = 'inside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(mean_result_name % suff))
        fig, ax = plotter.plot_with_error(
            preplot=(fig, ax), xlabel='time (s)', ylabel='Mean path efficiency', title='', c='navy', label=suff)

        ax.axvline(0, ls='--', c='k')
        ax.set_xlim((-2, 8))
        plotter.save(fig)

    def compute_mean_w10s_food_path_efficiency_around_attachments(self, redo):

        variable_name = 'w10s_food_path_efficiency_around_%s_attachments'
        mean_result_name = 'mean_' + variable_name
        hist_result_name = 'histograms_' + variable_name

        mean_label = 'Mean of the food path efficiency around %s attachments'
        mean_description = 'Mean of the food path efficiency for each time t in time_intervals,' \
                           ' which are times around %s attachments'

        hist_label = 'Histograms of the food path efficiency around %s attachments'
        hist_description = 'Histograms of the food path efficiency for each time t in time_intervals,' \
                           ' which are times around %s attachments'

        for suff in ['outside', 'inside']:
            self.__compute_mean_around_attachments(
                self.__compute_mean, variable_name % suff, mean_result_name % suff, hist_result_name % suff,
                mean_label % suff, mean_description % suff,
                hist_label % suff, hist_description % suff, redo)

        suff = 'outside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(hist_result_name % suff))
        fig, ax = plotter.create_plot(figsize=(9, 5), ncols=2, left=0.05)
        plotter.plot(preplot=(fig, ax[0]), title=suff, xlabel='Path Efficiency', ylabel='PDF', normed=True)
        ax[0].set_xlim(0, 1)
        suff = 'inside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(hist_result_name % suff))
        plotter.plot(preplot=(fig, ax[1]), title=suff, xlabel='Path Efficiency', ylabel='PDF', normed=True)
        ax[1].set_xlim(0, 1)
        plotter.save(fig, name=hist_result_name % '')

        suff = 'outside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(mean_result_name % suff))
        fig, ax = plotter.plot_with_error(
            xlabel='time (s)', ylabel='Mean path efficiency', title='', c='red', label=suff)

        suff = 'inside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(mean_result_name % suff))
        fig, ax = plotter.plot_with_error(
            preplot=(fig, ax), xlabel='time (s)', ylabel='Mean path efficiency', title='', c='navy', label=suff)

        ax.axvline(0, ls='--', c='k')
        ax.set_xlim((-2, 8))
        plotter.save(fig)

    def compute_mean_w10s_food_path_efficiency_around_leading_attachments(self, redo):

        variable_name = 'w10s_food_path_efficiency_around_%s_leading_attachments'
        mean_result_name = 'mean_' + variable_name
        hist_result_name = 'histograms_' + variable_name

        mean_label = 'Mean of the food path efficiency around %s leader attachments'
        mean_description = 'Mean of the food path efficiency for each time t in time_intervals,' \
                           ' which are times around leading %s attachments'

        hist_label = 'Histograms of the food path efficiency around %s leader attachments'
        hist_description = 'Histograms of the food path efficiency for each time t in time_intervals,' \
                           ' which are times around leading %s attachments'

        for suff in ['outside', 'inside']:
            self.__compute_mean_around_attachments(
                self.__compute_mean, variable_name % suff, mean_result_name % suff, hist_result_name % suff,
                mean_label % suff, mean_description % suff,
                hist_label % suff, hist_description % suff, redo)

        suff = 'outside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(hist_result_name % suff))
        fig, ax = plotter.create_plot(figsize=(9, 5), ncols=2, left=0.05)
        plotter.plot(preplot=(fig, ax[0]), title=suff, xlabel='Path Efficiency', ylabel='PDF', normed=True)
        ax[0].set_xlim(0, 1)
        suff = 'inside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(hist_result_name % suff))
        plotter.plot(preplot=(fig, ax[1]), title=suff, xlabel='Path Efficiency', ylabel='PDF', normed=True)
        ax[1].set_xlim(0, 1)
        plotter.save(fig, name=hist_result_name % '')

        suff = 'outside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(mean_result_name % suff))
        fig, ax = plotter.plot_with_error(
            xlabel='time (s)', ylabel='Mean path efficiency', title='', c='red', label=suff)

        suff = 'inside'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(mean_result_name % suff))
        fig, ax = plotter.plot_with_error(
            preplot=(fig, ax), xlabel='time (s)', ylabel='Mean path efficiency', title='', c='navy', label=suff)

        ax.axvline(0, ls='--', c='k')
        ax.set_xlim((-2, 8))
        plotter.save(fig)

    def compute_w1s_food_path_efficiency_around_attachments2(self):
        variable_name = 'w1s_food_path_efficiency'
        discrim_name = 'w10s_food_path_efficiency'
        attachment_name = '%s_attachment_intervals'
        result_name = variable_name + '_around_%s_attachments2'

        result_label = 'Food path efficiency around %s attachments'
        result_description = 'Food path efficiency with window 1s' \
                             ' before and after an ant coming from %s ant attached to the food'
        for suff in ['outside', 'inside']:

            self.__gather_exp_frame_indexed_around_attachments_with_discrimination(
                variable_name, discrim_name, attachment_name % suff, result_name % suff,
                result_label % suff, result_description % suff)

    def compute_w1s_food_path_efficiency_around_leading_attachments2(self):
        variable_name = 'w1s_food_path_efficiency'
        discrim_name = 'w1s_food_path_efficiency'
        attachment_name = '%s_leading_attachment_intervals'
        result_name = variable_name + '_around_%s_leading_attachments2'

        result_label = 'Food path efficiency around %s leading attachments'
        result_description = 'Food path efficiency with window 1s' \
                             ' before and after an ant coming from %s ant attached to the food and has' \
                             ' an influence on it'
        for suff in ['outside', 'inside']:

            self.__gather_exp_frame_indexed_around_attachments_with_discrimination(
                variable_name, discrim_name, attachment_name % suff, result_name % suff,
                result_label % suff, result_description % suff)

    def compute_w10s_food_path_efficiency_around_attachments2(self):
        variable_name = 'w10s_food_path_efficiency'
        discrim_name = 'w10s_food_path_efficiency'
        attachment_name = '%s_attachment_intervals'
        result_name = variable_name + '_around_%s_attachments2'

        result_label = 'Food path efficiency around %s attachments'
        result_description = 'Food path efficiency with window 1s' \
                             ' before and after an ant coming from %s ant attached to the food'
        for suff in ['outside', 'inside']:

            self.__gather_exp_frame_indexed_around_attachments_with_discrimination(
                variable_name, discrim_name, attachment_name % suff, result_name % suff,
                result_label % suff, result_description % suff)

    def compute_w10s_food_path_efficiency_around_leading_attachments2(self):
        variable_name = 'w10s_food_path_efficiency'
        discrim_name = 'w10s_food_path_efficiency'
        attachment_name = '%s_leading_attachment_intervals'
        result_name = variable_name + '_around_%s_leading_attachments2'

        result_label = 'Food path efficiency around %s leading attachments'
        result_description = 'Food path efficiency with window 1s' \
                             ' before and after an ant coming from %s ant attached to the food and has' \
                             ' an influence on it'
        for suff in ['outside', 'inside']:

            self.__gather_exp_frame_indexed_around_attachments_with_discrimination(
                variable_name, discrim_name, attachment_name % suff, result_name % suff,
                result_label % suff, result_description % suff)

    def compute_mean_w1s_food_path_efficiency_around_leading_attachments2(self, redo):

        variable_name = 'w1s_food_path_efficiency_around_%s_leading_attachments2'
        mean_result_name = 'mean_' + variable_name
        hist_result_name = 'histograms_' + variable_name

        mean_label = 'Mean of the food path efficiency around %s leader attachments'
        mean_description = 'Mean of the food path efficiency for each time t in time_intervals,' \
                           ' which are times around leading %s attachments'

        hist_label = 'Histograms of the food path efficiency around %s leader attachments'
        hist_description = 'Histograms of the food path efficiency for each time t in time_intervals,' \
                           ' which are times around leading %s attachments'

        dx = 0.1
        a0, a1, da = 0, 1+dx, dx
        eff_intervals = np.around(np.arange(a0, a1, da), 1)

        for suff in ['outside', 'inside']:
            self.__compute_mean_around_attachments2(
                self.__compute_mean2, eff_intervals, eff_intervals,
                variable_name % suff, mean_result_name % suff, hist_result_name % suff,
                mean_label % suff, mean_description % suff,
                hist_label % suff, hist_description % suff, redo)

        self._plot_food_path_efficiency_around_attachments(suff, mean_result_name, hist_result_name, eff_intervals)

    def compute_mean_w10s_food_path_efficiency_around_attachments2(self, redo):

        variable_name = 'w10s_food_path_efficiency_around_%s_attachments2'
        mean_result_name = 'mean_' + variable_name
        hist_result_name = 'histograms_' + variable_name

        mean_label = 'Mean of the food path efficiency around %s attachments'
        mean_description = 'Mean of the food path efficiency for each time t in time_intervals,' \
                           ' which are times around leading %s attachments'

        hist_label = 'Histograms of the food path efficiency around %s attachments'
        hist_description = 'Histograms of the food path efficiency for each time t in time_intervals,' \
                           ' which are times around leading %s attachments'

        dx = 0.1
        a0, a1, da = 0, 1+dx, dx
        eff_intervals = np.around(np.arange(a0, a1, da), 1)

        for suff in ['outside', 'inside']:
            self.__compute_mean_around_attachments2(
                self.__compute_mean2, eff_intervals, eff_intervals,
                variable_name % suff, mean_result_name % suff, hist_result_name % suff,
                mean_label % suff, mean_description % suff,
                hist_label % suff, hist_description % suff, redo)

        self._plot_food_path_efficiency_around_attachments(suff, mean_result_name, hist_result_name, eff_intervals)

    def compute_mean_w10s_food_path_efficiency_around_leading_attachments2(self, redo):

        variable_name = 'w10s_food_path_efficiency_around_%s_leading_attachments2'
        mean_result_name = 'mean_' + variable_name
        hist_result_name = 'histograms_' + variable_name

        mean_label = 'Mean of the food path efficiency around %s leader attachments'
        mean_description = 'Mean of the food path efficiency for each time t in time_intervals,' \
                           ' which are times around leading %s attachments'

        hist_label = 'Histograms of the food path efficiency around %s leader attachments'
        hist_description = 'Histograms of the food path efficiency for each time t in time_intervals,' \
                           ' which are times around leading %s attachments'

        dx = 0.1
        a0, a1, da = 0, 1+dx, dx
        eff_intervals = np.around(np.arange(a0, a1, da), 1)

        for suff in ['outside', 'inside']:
            self.__compute_mean_around_attachments2(
                self.__compute_mean2, eff_intervals, eff_intervals,
                variable_name % suff, mean_result_name % suff, hist_result_name % suff,
                mean_label % suff, mean_description % suff,
                hist_label % suff, hist_description % suff, redo)

        self._plot_food_path_efficiency_around_attachments(suff, mean_result_name, hist_result_name, eff_intervals)

    def _plot_food_path_efficiency_around_attachments(
            self, suff, mean_result_name, hist_result_name, eff_intervals):

        temp_name = 'temp_name'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(hist_result_name % suff))
        fig, ax = plotter.create_plot(
            figsize=(10, 14), nrows=5, ncols=4,
            left=0.04, bottom=0.04, top=0.96, hspace=0.6, wspace=0.3)

        title = '%s\n' + r'path eff. $\in$[%.2f, %.2f]'
        for k, eff in enumerate(eff_intervals[:-1]):
            eff1 = eff_intervals[k + 1]
            if k < 5:
                j = int(k / 5)
                i = k - j * 5
            else:
                j = int((k - 5) / 5) + 2
                i = (k - 5) - (j - 2) * 5

            print(k, i, j)

            suff = 'outside'
            df = self.exp.get_df(hist_result_name % suff).loc[eff, :]
            self.exp.add_new_dataset_from_df(df, temp_name, category=self.category, replace=True)
            plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(temp_name))
            plotter.plot(preplot=(fig, ax[i, j]), xlabel='Path Efficiency', ylabel='PDF', normed=True)
            ax[i, j].set_xlim(0, 1)
            ax[i, j].set_title(title % (suff, eff, eff1), color='r')

            suff = 'inside'
            df = self.exp.get_df(hist_result_name % suff).loc[eff, :]
            self.exp.add_new_dataset_from_df(df, temp_name, category=self.category, replace=True)
            plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(temp_name))
            plotter.plot(preplot=(fig, ax[i, j + 1]), xlabel='Path Efficiency', ylabel='PDF', normed=True)
            ax[i, j + 1].set_xlim(0, 1)
            ax[i, j + 1].set_title(title % (suff, eff, eff1), color='navy')

        plotter.save(fig, name=hist_result_name % '')
        temp_name = 'temp_name'

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(mean_result_name % suff))
        ncols = 5
        fig, ax = plotter.create_plot(
            figsize=(6.5, 15), nrows=5, ncols=2,
            left=0.09, bottom=0.05, top=0.97, right=0.99, wspace=0.3, hspace=0.4)
        title = r'path eff. $\in$[%.2f, %.2f]'

        for k, eff in enumerate(eff_intervals[:-1]):
            eff1 = eff_intervals[k + 1]
            j = int(k / ncols)
            i = k - j * ncols
            suff = 'outside'
            df = self.exp.get_df(mean_result_name % suff).loc[eff, :]
            self.exp.add_new_dataset_from_df(df, temp_name, category=self.category, replace=True)
            plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(temp_name))
            plotter.plot_with_error(
                preplot=(fig, ax[i, j]), xlabel='time (s)',
                ylabel='Mean path efficiency', title=title % (eff, eff1), c='red', label=suff)

            suff = 'inside'
            df = self.exp.get_df(mean_result_name % suff).loc[eff, :]
            self.exp.add_new_dataset_from_df(df, temp_name, category=self.category, replace=True)
            plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(temp_name))
            plotter.plot_with_error(
                preplot=(fig, ax[i, j]), xlabel='time (s)',
                ylabel='Mean path efficiency', title=title % (eff, eff1), c='navy', label=suff)

            plotter.draw_vertical_line(ax[i, j])
            plotter.draw_horizontal_line(ax[i, j], eff, c='w')
            plotter.draw_horizontal_line(ax[i, j], eff1, c='w')
            ax[i, j].set_xlim(-2, 8)
            ax[i, j].set_ylim(0, 1)

        plotter.save(fig, name=mean_result_name % '')

    def __compute_mean_around_attachments2(
            self, fct, eff_intervals, eff_intervals2, variable_name, mean_result_name, hist_result_name,
            mean_label, mean_description, hist_label, hist_description, redo, list_exps=None):

        t0, t1, dt = -2, 8, 2.
        time_intervals = np.around(np.arange(t0, t1+1 + dt, dt), 1)

        bins = np.arange(0, 1.1, 0.1)
        hists_index_values = np.around((bins[1:] + bins[:-1]) / 2., 3)

        index_values = [(a, h) for a in eff_intervals[:-1] for h in hists_index_values]

        if redo:
            self.exp.load(variable_name)

            self.exp.add_new_empty_dataset(name=hist_result_name,
                                           index_names=['food_path_efficiency2', 'food_path_efficiency'],
                                           column_names=time_intervals[:-1], index_values=index_values,
                                           category=self.category, label=hist_label, description=hist_description,
                                           replace=True)

            for i in range(len(time_intervals)-1):
                t0 = str(time_intervals[i])
                t1 = str(time_intervals[i+1])
                for j in range(len(eff_intervals)-1):
                    eff0 = eff_intervals[j]
                    eff1 = eff_intervals[j+1]

                    values = self.exp.get_df(variable_name).loc[pd.IndexSlice[:, :, eff0:eff1], t0:t1].dropna()
                    if list_exps is not None:
                        values = values.loc[list_exps, :]
                    values = values.values.ravel()
                    hist = np.histogram(values, bins=bins, normed=False)[0]

                    self.exp.get_df(hist_result_name).loc[pd.IndexSlice[eff0, :], float(t0)] = hist

            t0, t1, dt = -10., 10., .5
            time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)

            index_values = [(a, h) for a in eff_intervals2[:-1] for h in time_intervals]

            self.exp.add_new_empty_dataset(name=mean_result_name, index_names=['food_path_efficiency2', 'time'],
                                           column_names=['mean', 'CI95_1', 'CI95_2'],
                                           index_values=index_values, category=self.category,
                                           label=mean_label, description=mean_description, replace=True)

            fct(time_intervals, eff_intervals2, variable_name, mean_result_name)

            self.exp.remove_object(variable_name)
            self.exp.write(hist_result_name)
            self.exp.write(mean_result_name)
        else:
            self.exp.load(mean_result_name)
            self.exp.load(hist_result_name)

    def compute_w10s_food_path_efficiency_mean_evol_around_first_outside_attachment(self, redo=False):

        name = 'w10s_food_path_efficiency'
        result_name = name + '_mean_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.1
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(-1, 4., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Mean of the food path efficiency over time'
        description = 'Mean of the food path efficiency over time'

        if redo:

            self.exp.load(name)

            self.change_first_frame(name, init_frame_name)

            self.exp.mean_evolution(name_to_var=name, start_index_intervals=start_frame_intervals,
                                    end_index_intervals=end_frame_intervals, error=True,
                                    category=self.category, result_name=result_name,
                                    label=label, description=description, replace=True)
            self.exp.write(result_name)
            self.exp.remove_object(name)

        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot_with_error(
            xlabel='Time (s)', ylabel='Mean', label_suffix='s', label='Mean', marker='')
        ax.set_ylim((0, 1))
        plotter.save(fig)
