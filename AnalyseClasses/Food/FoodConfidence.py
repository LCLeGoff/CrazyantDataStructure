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

    def compute_w10s_food_crossed_distance(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 10
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

    def compute_w10s_food_total_crossed_distance(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 10
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
                w_in_f2 = int(w_in_f / 2)

                df2 = df.loc[id_exp, :]

                xy1 = df2.loc[frame0 + 1:]
                xy2 = df2.loc[:frame1-1]
                xy1.index -= 1
                xy2.index += 1
                dist = distance_df(xy1, xy2)*fps/2.
                dist2 = dist.copy()

                for frame in dist.index:
                    dist2.loc[frame] = np.nanmean(dist.loc[frame-w_in_f2:frame+w_in_f2])

                dist2 = dist2.reindex(df.index.get_level_values(id_frame_name))
                self.exp.get_df(result_name).loc[id_exp, :] = np.array(np.around(dist2, 5))

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
            attachment_name = 'outside_ant_carrying_intervals'
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

    def compute_w10s_food_path_efficiency(self, redo=False, redo_hist=False, redo_plot_indiv=False):
        w = 10
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
                      'divided by the total distance crossed by  te food during the same time'
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
        fig, ax = plotter.plot()
        plotter.save(fig)
