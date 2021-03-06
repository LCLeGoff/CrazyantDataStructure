import numpy as np
import pandas as pd

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_frame_name
from Tools.MiscellaneousTools.ArrayManipulation import get_entropy, get_max_entropy, get_interval_containing
from Tools.MiscellaneousTools.FoodFisherInformation import compute_fisher_information_uniform_von_mises
from Tools.Plotter.ColorObject import ColorObject
from Tools.Plotter.Plotter import Plotter


class AnalyseFoodInformation(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'FoodInformation'

    def compute_w1s_entropy_mm1s_food_velocity_phi_indiv_evol(self, redo=False, redo_indiv_plot=False):
        mm = 1
        w = 1

        result_name = 'w'+str(w)+'s_entropy_mm1s_food_velocity_phi_indiv_evol'

        time_intervals = np.arange(0, 10*60, 1)
        dtheta = np.pi / 12.
        bins = np.around(np.arange(-np.pi + dtheta/2., np.pi + dtheta, dtheta), 3)

        self.__compute_food_velocity_entropy(
            w, bins, self.category, mm, redo, redo_indiv_plot, result_name, time_intervals)

    def compute_w10s_entropy_mm1s_food_velocity_phi_indiv_evol(self, redo=False, redo_indiv_plot=False):
        mm = 1
        w = 10

        result_name = 'w'+str(w)+'s_entropy_mm1s_food_velocity_phi_indiv_evol'

        time_intervals = np.arange(0, 10*60, 1)
        dtheta = np.pi / 12.
        bins = np.around(np.arange(-np.pi + dtheta/2., np.pi + dtheta, dtheta), 3)

        self.__compute_food_velocity_entropy(
            w, bins, self.category, mm, redo, redo_indiv_plot, result_name, time_intervals)

    def compute_w30s_entropy_mm1s_food_velocity_phi_indiv_evol(self, redo=False, redo_indiv_plot=False):
        mm = 1
        w = 30

        result_name = 'w'+str(w)+'s_entropy_mm1s_food_velocity_phi_indiv_evol'

        time_intervals = np.arange(0, 10*60)
        dtheta = np.pi / 12.
        bins = np.around(np.arange(-np.pi + dtheta/2., np.pi + dtheta, dtheta), 3)

        self.__compute_food_velocity_entropy(
            w, bins, self.category, mm, redo, redo_indiv_plot, result_name, time_intervals)

    def __compute_food_velocity_entropy(self,
                                        w, bins, category, mm, redo, redo_indiv_plot, result_name, time_intervals):
        if redo:
            vel_phi_name = 'mm' + str(mm) + 's_food_velocity_phi'
            self.exp.load([vel_phi_name, 'fps'])
            self.exp.add_new_empty_dataset(name=result_name, index_names='time',
                                           column_names=self.exp.id_exp_list, index_values=time_intervals,
                                           category=category, label='Evolution of the entropy of the food velocity phi',
                                           description='Time evolution of the entropy of the distribution'
                                                       ' of the angular coordinate'
                                                       ' of food velocity for each experiment')

            def compute_entropy4each_group(df: pd.DataFrame):
                exp = df.index.get_level_values(id_exp_name)[0]
                print(exp)
                fps0 = self.exp.get_value('fps', exp)
                frame0 = df.index.get_level_values(id_frame_name).min()

                w0 = w * fps0
                for time in time_intervals:
                    f0 = time * fps0 - w0 + frame0
                    f1 = time * fps0 + w0 + frame0

                    vel = df.loc[pd.IndexSlice[exp, f0:f1], :]
                    hist = np.histogram(vel, bins, normed=False)
                    hist = hist[0] / np.sum(hist[0])
                    if len(vel) != 0:
                        entropy = np.around(get_entropy(hist), 3)
                        self.exp.change_value(result_name, (time, exp), entropy)

            self.exp.groupby(vel_phi_name, id_exp_name, compute_entropy4each_group)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        if redo or redo_indiv_plot:

            attachment_name = 'outside_ant_carrying_intervals'
            self.exp.load(['fps', attachment_name])

            for id_exp in self.exp.get_df(result_name).columns:
                id_exp = int(id_exp)
                fps = self.exp.get_value('fps', id_exp)

                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name=id_exp)
                fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Entropy',
                                       title_prefix='Exp ' + str(id_exp) + ': ')

                attachments = self.exp.get_df(attachment_name).loc[id_exp, :]
                attachments.reset_index(inplace=True)
                attachments = np.array(attachments)

                colors = plotter.color_object.create_cmap('hot_r', set(list(attachments[:, 0])))
                for id_ant, frame, inter in attachments:
                    ax.axvline(frame / fps, c=colors[str(id_ant)], alpha=0.5)
                ax.grid()
                ax.set_ylim((0.5, 4))
                plotter.save(fig, name=id_exp, sub_folder=result_name)

    def compute_mm1s_food_direction_error_around_outside_attachments(self):
        mm = 1
        attachment_name = 'outside_ant_carrying_intervals'
        self.exp.load(attachment_name)
        variable_name = 'mm' + str(mm) + 's_food_direction_error'

        result_name = variable_name + '_around_outside_attachments'

        result_label = 'Food direction error around outside attachments'
        result_description = 'Food direction error smoothed with a moving mean of window ' + str(mm) + ' s for times' \
                             ' before and after an ant coming from outside ant attached to the food'

        self.__gather_exp_frame_indexed_around_attachments(variable_name, attachment_name, result_name, result_label,
                                                           result_description)

    def compute_mm1s_food_direction_error_around_non_outside_attachments(self):
        mm = 1
        attachment_name = 'non_outside_ant_carrying_intervals'
        self.exp.load(attachment_name)
        variable_name = 'mm' + str(mm) + 's_food_direction_error'

        result_name = variable_name + '_around_non_outside_attachments'

        result_label = 'Food direction error around non outside attachments'
        result_description = 'Food direction error smoothed with a moving mean of window ' + str(mm) + ' s for times' \
                             ' before and after an ant coming from non outside ant attached to the food'

        self.__gather_exp_frame_indexed_around_attachments(variable_name, attachment_name, result_name, result_label,
                                                           result_description)

    def compute_mm1s_food_direction_error_around_non_outside_attachments_after_first_outside_attachment(self):
        mm = 1
        attachment_name = 'non_outside_ant_carrying_intervals'
        self.exp.load(attachment_name)
        variable_name = 'mm' + str(mm) + 's_food_direction_error'

        result_name = variable_name + '_around_non_outside_attachments_after_first_outside_attachment'

        result_label = 'Food direction error around non outside attachments after first outside attachment'
        result_description = 'Food direction error smoothed with a moving mean of window ' + str(mm) + ' s for times' \
                             ' before and after an ant coming from non outside ant attached to the food'

        self.__gather_variable_around_attachments_after_first_outside_attachment(
            variable_name, attachment_name, result_name, result_label, result_description)

    def compute_mm1s_food_direction_error_around_attachments_after_first_outside_attachment(self):
        mm = 1
        attachment_name = 'carrying_intervals'
        self.exp.load(attachment_name)
        variable_name = 'mm' + str(mm) + 's_food_direction_error'

        result_name = variable_name + '_around_attachments_after_first_outside_attachment'

        result_label = 'Food direction error around non outside attachments after first outside attachment'
        result_description = 'Food direction error smoothed with a moving mean of window ' + str(mm) + ' s for times' \
                             ' before and after an ant coming from non outside ant attached to the food'

        self.__gather_variable_around_attachments_after_first_outside_attachment(
            variable_name, attachment_name, result_name, result_label, result_description)

    def compute_mm1s_food_direction_error_around_isolated_outside_attachments(self):
        mm = 1
        attachment_name = 'isolated_outside_ant_carrying_intervals'
        self.exp.load(attachment_name)
        variable_name = 'mm' + str(mm) + 's_food_direction_error'

        result_name = variable_name + '_around_isolated_outside_attachments'

        result_label = 'Food direction error around isolated outside attachments'
        result_description = 'Food direction error smoothed with a moving mean of window ' + str(mm) + ' s for times' \
                             ' before and after an ant coming from outside ant attached to the food,' \
                             'moreover no outside ant attachments occurs during this period of time'

        self.__gather_exp_frame_indexed_around_attachments(variable_name, attachment_name, result_name, result_label,
                                                           result_description)

    def compute_mm1s_food_direction_error_around_isolated_non_outside_attachments(self):
        mm = 1
        attachment_name = 'isolated_non_outside_ant_carrying_intervals'
        self.exp.load(attachment_name)
        variable_name = 'mm' + str(mm) + 's_food_direction_error'

        result_name = variable_name + '_around_isolated_non_outside_attachments'

        result_label = 'Food direction error around isolated non outside attachments'
        result_description = 'Food direction error smoothed with a moving mean of window ' + str(mm) + ' s for times' \
                             ' before and after an ant coming from non outside ant attached to the food,' \
                             'moreover no outside ant attachments occurs during this period of time'

        self.__gather_exp_frame_indexed_around_attachments(variable_name, attachment_name, result_name, result_label,
                                                           result_description)

    def __gather_exp_frame_indexed_around_attachments(self, variable_name, attachment_name, result_name, result_label,
                                                      result_description):

        last_frame_name = 'food_exit_frames'
        self.exp.load([variable_name, last_frame_name, 'fps'])

        t0, t1, dt = -60, 60, 0.1
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)

        index_values = self.exp.get_df(attachment_name).reset_index()
        index_values = index_values.set_index([id_exp_name, id_frame_name])
        index_values = index_values.sort_index()
        index_values = index_values.index
        self.exp.add_new_empty_dataset(name=result_name, index_names=[id_exp_name, id_frame_name],
                                       column_names=np.array(time_intervals, dtype=str),
                                       index_values=index_values,
                                       category=self.category, label=result_label, description=result_description)

        def get_variable4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            fps = self.exp.get_value('fps', id_exp)
            last_frame = self.exp.get_value(last_frame_name, id_exp)

            if id_exp in self.exp.get_index(attachment_name).get_level_values(id_exp_name):
                attachment_frames = self.exp.get_df(attachment_name).loc[id_exp, :, :]
                attachment_frames = list(set(attachment_frames.index.get_level_values(id_frame_name)))
                attachment_frames.sort()

                for attach_frame in attachment_frames:
                    if attach_frame < last_frame:
                        print(id_exp, attach_frame)
                        f0 = int(attach_frame + time_intervals[0] * fps)
                        f1 = int(attach_frame + time_intervals[-1] * fps)

                        var_df = df.loc[pd.IndexSlice[id_exp, f0:f1], :]
                        var_df = var_df.loc[id_exp, :]
                        var_df.index -= attach_frame
                        var_df.index /= fps

                        var_df = var_df.reindex(time_intervals)

                        self.exp.get_df(result_name).loc[(id_exp, attach_frame), :] = np.array(var_df[variable_name])

        self.exp.groupby(variable_name, id_exp_name, func=get_variable4each_group)
        self.exp.write(result_name)

    def compute_mm1s_food_direction_error_around_outside_manual_leader_attachments(self):
        mm = 1
        attachment_name = 'outside_manual_leading_attachments'
        variable_name = 'mm' + str(mm) + 's_food_direction_error'

        result_name = variable_name + '_around_outside_manual_leading_attachments'

        self.exp.load(attachment_name)
        df = self.exp.get_df(attachment_name).copy()
        df = df[df == 1].dropna()
        self.exp.change_df(attachment_name, df)

        result_label = 'Food direction error around manual leading outside attachments'
        result_description = 'Food direction error smoothed with a moving mean of window ' + str(mm) + ' s for times' \
                             ' before and after an ant coming from outside ant attached to the food,' \
                             ' the attachments have been manually identified has leading'

        self.__gather_exp_frame_indexed_around_attachments(variable_name, attachment_name, result_name, result_label,
                                                           result_description)
        self.exp.remove_object(attachment_name)

    def compute_mm1s_food_direction_error_around_inside_manual_leader_attachments(self):
        mm = 1
        attachment_name = 'inside_manual_leading_attachments'
        variable_name = 'mm' + str(mm) + 's_food_direction_error'

        result_name = variable_name + '_around_inside_manual_leading_attachments'

        self.exp.load(attachment_name)
        df = self.exp.get_df(attachment_name).copy()
        df = df[df == 1].dropna()
        self.exp.change_df(attachment_name, df)

        result_label = 'Food direction error around manual leading inside attachments'
        result_description = 'Food direction error smoothed with a moving mean of window ' + str(mm) + ' s for times' \
                             ' before and after an ant coming from inside ant attached to the food,' \
                             ' the attachments have been manually identified has leading'

        self.__gather_exp_frame_indexed_around_attachments(variable_name, attachment_name, result_name, result_label,
                                                           result_description)
        self.exp.remove_object(attachment_name)

    def compute_mm1s_food_direction_error_around_outside_manual_follower_attachments(self):
        mm = 1
        attachment_name = 'outside_manual_leading_attachments'
        variable_name = 'mm' + str(mm) + 's_food_direction_error'

        result_name = variable_name + '_around_outside_manual_following_attachments'

        self.exp.load(attachment_name)
        df = self.exp.get_df(attachment_name).copy()
        df = df[df == 0].dropna()
        self.exp.change_df(attachment_name, df)

        result_label = 'Food direction error around manual following outside attachments'
        result_description = 'Food direction error smoothed with a moving mean of window ' + str(mm) + ' s for times' \
                             ' before and after an ant coming from outside ant attached to the food,' \
                             ' the attachments have been manually identified has following'

        self.__gather_exp_frame_indexed_around_attachments(variable_name, attachment_name, result_name, result_label,
                                                           result_description)
        self.exp.remove_object(attachment_name)

    def compute_mm1s_food_direction_error_around_inside_manual_follower_attachments(self):
        mm = 1
        attachment_name = 'inside_manual_leading_attachments'
        variable_name = 'mm' + str(mm) + 's_food_direction_error'

        result_name = variable_name + '_around_inside_manual_following_attachments'
        self.exp.load(attachment_name)

        df = self.exp.get_df(attachment_name).copy()
        df = df[df == 0].dropna()
        self.exp.change_df(attachment_name, df)

        result_label = 'Food direction error around manual following inside attachments'
        result_description = 'Food direction error smoothed with a moving mean of window ' + str(mm) + ' s for times' \
                             ' before and after an ant coming from inside ant attached to the food,' \
                             ' the attachments have been manually identified has following'

        self.__gather_exp_frame_indexed_around_attachments(variable_name, attachment_name, result_name, result_label,
                                                           result_description)
        self.exp.remove_object(attachment_name)

    def __gather_variable_around_attachments_after_first_outside_attachment(
            self, variable_name, attachment_name, result_name, result_label, result_description):

        last_frame_name = 'food_exit_frames'
        first_frame_name = 'first_attachment_time_of_outside_ant'
        self.exp.load([attachment_name, variable_name, first_frame_name, last_frame_name, 'fps'])

        t0, t1, dt = -60, 60, 0.1
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)

        index_names = self.exp.get_df(attachment_name).reset_index()
        index_names = index_names.set_index([id_exp_name, id_frame_name])
        index_names = index_names.sort_index()
        index_names = index_names.index
        self.exp.add_new_empty_dataset(name=result_name, index_names=[id_exp_name, id_frame_name],
                                       column_names=np.array(time_intervals, dtype=str),
                                       index_values=index_names,
                                       category=self.category, label=result_label, description=result_description)

        def get_variable4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            fps = self.exp.get_value('fps', id_exp)
            last_frame = self.exp.get_value(last_frame_name, id_exp)
            first_frame = self.exp.get_value(first_frame_name, id_exp)

            if id_exp in self.exp.get_index(attachment_name).get_level_values(id_exp_name):
                attachment_frames = self.exp.get_df(attachment_name).loc[id_exp, :]
                attachment_frames = list(set(attachment_frames.index.get_level_values(id_frame_name)))
                attachment_frames.sort()

                for attach_frame in attachment_frames:
                    if first_frame <= attach_frame < last_frame:
                        print(id_exp, attach_frame)
                        f0 = int(attach_frame + time_intervals[0] * fps)
                        f1 = int(attach_frame + time_intervals[-1] * fps)

                        var_df = df.loc[pd.IndexSlice[id_exp, f0:f1], :]
                        var_df = var_df.loc[id_exp, :]
                        var_df.index -= attach_frame
                        var_df.index /= fps

                        var_df = var_df.reindex(time_intervals)

                        self.exp.get_df(result_name).loc[(id_exp, attach_frame), :] = np.array(var_df[variable_name])

        self.exp.groupby(variable_name, id_exp_name, func=get_variable4each_group)
        self.exp.write(result_name)

    def compute_information_mm1s_food_direction_error_around_outside_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'mm1s_food_direction_error_around_outside_attachments'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around outside attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around outside ant attachments'

        info_label = 'Information of the food around outside attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside ant attachments'

        ylim_zoom = (0.2, 0.55)
        dpi = 1/12.
        self.__compute_information_around_attachments(self.__compute_entropy, dpi, variable_name, hists_result_name,
                                                      info_result_name, hists_label, hists_description,
                                                      info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

    def compute_information_mm1s_food_direction_error_around_isolated_outside_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'mm1s_food_direction_error_around_isolated_outside_attachments'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around isolated outside attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around isolated outside ant attachments'

        info_label = 'Information of the food around isolated outside attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around isolated outside ant attachments'

        ylim_zoom = (0., 0.4)
        dpi = 1/12.
        self.__compute_information_around_attachments(self.__compute_entropy, dpi, variable_name, hists_result_name,
                                                      info_result_name, hists_label, hists_description,
                                                      info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

    def compute_information_mm1s_food_direction_error_around_isolated_non_outside_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'mm1s_food_direction_error_around_isolated_non_outside_attachments'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around isolated non outside attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around isolated non outside ant attachments'

        info_label = 'Information of the food around isolated non outside attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals,' \
                           ' which are times around isolated non outside ant attachments'

        ylim_zoom = (0., 0.4)
        dpi = 1/6.
        self.__compute_information_around_attachments(self.__compute_entropy, dpi, variable_name, hists_result_name,
                                                      info_result_name, hists_label, hists_description,
                                                      info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

    def compute_information_mm1s_food_direction_error_around_non_outside_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'mm1s_food_direction_error_around_non_outside_attachments'

        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around non outside attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around non outside ant attachments'

        info_label = 'Information of the food around non outside attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around non outside ant attachments'

        ylim_zoom = (0.1, 0.2)
        dpi = 1/12.
        self.__compute_information_around_attachments(self.__compute_entropy, dpi, variable_name, hists_result_name,
                                                      info_result_name, hists_label, hists_description,
                                                      info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

    def compute_information_mm1s_food_direction_error_around_attachments_after_first_outside_attachment(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'mm1s_food_direction_error_around_attachments_after_first_outside_attachment'

        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around ant attachments'

        info_label = 'Information of the food around attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around ant attachments'

        ylim_zoom = (0.1, 0.2)
        dpi = 1/12.
        self.__compute_information_around_attachments(self.__compute_entropy, dpi, variable_name, hists_result_name,
                                                      info_result_name, hists_label, hists_description,
                                                      info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

    def compute_information_mm1s_food_direction_error_around_non_outside_attachments_after_first_outside_attachment(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'mm1s_food_direction_error_around_non_outside_attachments_after_first_outside_attachment'

        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around non outside attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around non outside ant attachments'

        info_label = 'Information of the food around outside attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around non outside ant attachments'

        ylim_zoom = (0.1, 0.2)
        dpi = 1/12.
        self.__compute_information_around_attachments(self.__compute_entropy, dpi, variable_name, hists_result_name,
                                                      info_result_name, hists_label, hists_description,
                                                      info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.plot(
            xlabel='time (s)', ylabel='Information (bit)', title='', label='inside attachments', c='w')

        variable_name = 'information_mm1s_food_direction_error_around_outside_attachments'
        self.exp.load(variable_name)
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(variable_name))
        plotter.plot(preplot=(fig, ax), xlabel='time (s)', ylabel='Information (bit)',
                     title='', c='r', label='outside attachments')

        variable_name = 'information_mm1s_food_direction_error_around_attachments_after_first_outside_attachment'
        self.exp.load(variable_name)
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(variable_name))
        plotter.plot(preplot=(fig, ax), xlabel='time (s)', ylabel='Information (bit)',
                     title='', label='all attachments', display_legend=True)

        ax.axvline(0, ls='--', c='k')
        ax.set_xlim((-2, 8))
        plotter.save(fig, name='information_mm1s_food_direction_error_around_outside_and_inside_attachments')

    def compute_information_mm1s_food_direction_error_around_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'mm1s_food_direction_error_around_outside_attachments'
        variable_name2 = 'mm1s_food_direction_error_around_non_outside_attachments'
        self.exp.load([variable_name, variable_name2])
        self.exp.get_data_object(variable_name).df =\
            pd.concat([self.exp.get_df(variable_name), self.exp.get_df(variable_name2)])

        hists_result_name = 'histograms_mm1s_food_direction_error_around_attachments'
        info_result_name = 'information_mm1s_food_direction_error_around_attachments'

        hists_label = 'Histograms of the food direction error around attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around ant attachments'

        info_label = 'Information of the food around outside attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around ant attachments'

        ylim_zoom = (0.15, 0.3)
        dpi = 1/12.
        self.__compute_information_around_attachments(self.__compute_entropy, dpi, variable_name, hists_result_name,
                                                      info_result_name, hists_label, hists_description,
                                                      info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

    def compute_information_mm1s_food_direction_error_around_outside_manual_leading_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'mm1s_food_direction_error_around_outside_manual_leading_attachments'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around manual outside attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around outside ant attachments,' \
                            ' the attachments have been manually identified has leading'

        info_label = 'Information of the food around outside attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside ant attachments,' \
                           ' the attachments have been manually identified has leading'

        ylim_zoom = (0.2, 1.2)
        dpi = 1/12.
        self.__compute_information_around_attachments(self.__compute_entropy, dpi, variable_name, hists_result_name,
                                                      info_result_name, hists_label, hists_description,
                                                      info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

    def compute_information_mm1s_food_direction_error_around_inside_manual_leading_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'mm1s_food_direction_error_around_inside_manual_leading_attachments'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around manual inside leading attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around inside ant attachments,' \
                            ' the attachments have been manually identified has leading'

        info_label = 'Information of the food around inside leading attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around inside ant attachments,' \
                           ' the attachments have been manually identified has leading'

        ylim_zoom = (0.2, 0.8)
        dpi = 1/12.
        self.__compute_information_around_attachments(self.__compute_entropy, dpi, variable_name, hists_result_name,
                                                      info_result_name, hists_label, hists_description,
                                                      info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

    def compute_information_mm1s_food_direction_error_around_outside_manual_following_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'mm1s_food_direction_error_around_outside_manual_following_attachments'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around manual following outside attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around outside ant attachments,' \
                            ' the attachments have been manually identified has following'

        info_label = 'Information of the food around outside following attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside ant attachments,' \
                           ' the attachments have been manually identified has following'

        ylim_zoom = (0.2, 0.8)
        dpi = 1/12.
        self.__compute_information_around_attachments(self.__compute_entropy, dpi, variable_name, hists_result_name,
                                                      info_result_name, hists_label, hists_description,
                                                      info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

    def compute_information_mm1s_food_direction_error_around_inside_manual_following_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'mm1s_food_direction_error_around_inside_manual_following_attachments'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around manual following inside attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around inside ant attachments,' \
                            ' the attachments have been manually identified has following'

        info_label = 'Information of the food around inside following attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around inside ant attachments,' \
                           ' the attachments have been manually identified has following'

        ylim_zoom = (0.1, 0.3)
        dpi = 1/12.
        self.__compute_information_around_attachments(self.__compute_entropy, dpi, variable_name, hists_result_name,
                                                      info_result_name, hists_label, hists_description,
                                                      info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

    def __compute_information_around_attachments(self, fct, dpi, variable_name, hists_result_name, info_result_name,
                                                 hists_label, hists_description, info_label, info_description,
                                                 ylim_zoom, redo, redo_info, redo_plot_hist):
        t0, t1, dt = -60, 60, 0.5
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)
        dtheta = np.pi * dpi
        bins = np.arange(0, np.pi + dtheta, dtheta)
        hists_index_values = np.around((bins[1:] + bins[:-1]) / 2., 3)
        if redo:

            self.exp.add_new_empty_dataset(name=hists_result_name, index_names='food_direction_error',
                                           column_names=time_intervals, index_values=hists_index_values,
                                           category=self.category, label=hists_label, description=hists_description)

            for t in time_intervals:
                values = self.exp.get_df(variable_name)[str(t)].dropna()
                hist = np.histogram(values, bins=bins, normed=False)[0]
                s = float(np.sum(hist))
                hist = hist / s

                self.exp.get_df(hists_result_name)[t] = hist

            self.exp.remove_object(variable_name)
            self.exp.write(hists_result_name)

        else:
            self.exp.load(hists_result_name)

        if redo or redo_info:
            time_intervals = self.exp.get_df(hists_result_name).columns

            self.exp.add_new_empty_dataset(name=info_result_name, index_names='time', column_names='info',
                                           index_values=time_intervals, category=self.category,
                                           label=info_label, description=info_description)

            fct(time_intervals, hists_result_name, info_result_name)

            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)

        ylabel = 'Information (bit)'
        ylim = (0, 0.25)

        self.__plot_info(hists_result_name, info_result_name, time_intervals, ylabel, ylim, ylim_zoom,
                         redo, redo_plot_hist)

    def __plot_info(
            self, hists_result_name, info_result_name, time_intervals, ylabel, ylim, ylim_zoom,
            redo, redo_plot_hist):

        if redo or redo_plot_hist:
            for t in time_intervals:
                plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(hists_result_name), column_name=t)
                fig, ax = plotter.plot(xlabel='Food direction error', ylabel='Probability')
                ax.set_ylim(ylim)
                plotter.save(fig, sub_folder=hists_result_name, name=t)

        self.exp.remove_object(info_result_name)
        self.exp.load(info_result_name)
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.create_plot(figsize=(4.5, 8), nrows=2)
        plotter.plot(preplot=(fig, ax[0]), xlabel='time (s)', ylabel=ylabel, title='')

        ax[0].axvline(0, ls='--', c='k')
        plotter.plot(preplot=(fig, ax[1]), title='')
        ax[1].axvline(0, ls='--', c='k')
        ax[1].set_xlim((-2, 8))
        ax[1].set_ylim(ylim_zoom)
        plotter.save(fig)

    def __plot_info_vs_nb_attach(self, info_result_name, xlabel, ylabel, ylim=None):

        self.exp.remove_object(info_result_name)
        self.exp.load(info_result_name)
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.plot(xlabel=xlabel, ylabel=ylabel, title='')
        if ylim is not None:
            ax.set_ylim(ylim)
        plotter.save(fig)

    def __compute_entropy(self, time_intervals, hists_result_name, info_result_name):
        for t in time_intervals:
            hist = self.exp.get_df(hists_result_name)[t]
            entropy = get_entropy(hist)
            max_entropy = get_max_entropy(hist)

            self.exp.get_df(info_result_name).loc[t] = np.around(max_entropy - entropy, 6)

    def compute_information_mm1s_food_direction_error_around_first_outside_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'mm1s_food_direction_error_around_outside_attachments'

        hists_result_name = 'histograms_mm1s_food_direction_error_around_first_outside_attachments'
        info_result_name = 'information_mm1s_food_direction_error_around_first_outside_attachments'

        hists_label = 'Histograms of the food direction error around first outside attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around first outside ant attachments'

        info_label = 'Information of the food around outside first outside attachments'
        info_description = 'Information of the food (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around first outside ant attachments'

        ylim = (.35, .75)
        self.__compute_information_around_outside_attachments(self.__compute_entropy_around_outside_attachments,
                                                              variable_name, hists_result_name, hists_label,
                                                              hists_description, info_result_name, info_label,
                                                              info_description, ylim, redo, redo_info, redo_plot_hist)

    def __compute_information_around_outside_attachments(self, fct, variable_name, hists_result_name, hists_label,
                                                         hists_description, info_result_name, info_label,
                                                         info_description, ylim, redo, redo_info, redo_plot_hist):
        self.exp.load([variable_name])
        rank_list = np.arange(1., 11.)
        t0, t1, dt = -60, 60, 0.1
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)
        t0, t1, dt = -60, 60, 1.
        time_intervals_to_plot_individually = np.around(np.arange(t0, t1 + dt, dt), 1)
        hists_index_values = [(th, t) for th in rank_list for t in time_intervals]
        dtheta = np.pi / 12.
        bins = np.arange(0, np.pi + dtheta, dtheta)
        bins2 = np.around((bins[1:] + bins[:-1]) / 2., 3)
        if redo:

            self.exp.add_new_empty_dataset(name=hists_result_name, index_names=['rank', 'time'],
                                           column_names=bins2, index_values=hists_index_values,
                                           category=self.category, label=hists_label, description=hists_description)

            first_attach_name = 'first_attachments'
            self.__extract_10first_attachments(first_attach_name, variable_name)

            for (th, t) in self.exp.get_index(hists_result_name):
                values = self.exp.get_df(first_attach_name).loc[pd.IndexSlice[:, th], str(t)].dropna()
                hist = np.histogram(values, bins=bins, density=False)[0]
                s = float(np.sum(hist))
                hist = hist / s

                self.exp.get_df(hists_result_name).loc[(th, t), :] = hist

            self.exp.write(hists_result_name)
        else:
            self.exp.load(hists_result_name)
        if redo or redo_info:

            self.exp.add_new_empty_dataset(name=info_result_name, index_names='time', column_names=rank_list,
                                           index_values=time_intervals, category=self.category,
                                           label=info_label, description=info_description)

            fct(hists_result_name, info_result_name)

            self.exp.remove_object(variable_name)
            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)
        if redo or redo_info or redo_plot_hist:
            for t in time_intervals_to_plot_individually:
                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hists_result_name))
                fig, ax = plotter.create_plot()
                colors = ColorObject.create_cmap('hot', rank_list)

                for th in rank_list:
                    df = self.exp.get_df(hists_result_name).loc[(th, t)]
                    df = pd.DataFrame(data=np.array(df), index=list(bins2), columns=['temp'])
                    self.exp.add_new_dataset_from_df(df=df, name='temp', category=self.category, replace=True)

                    plotter = Plotter(self.exp.root, obj=self.exp.get_data_object('temp'))
                    fig, ax = plotter.plot(preplot=(fig, ax), xlabel='Food direction error', ylabel='Probability',
                                           label=str(th) + 'th', c=colors[str(th)])

                ax.set_title(str(t) + ' (s)')
                ax.set_ylim((0, 0.4))
                ax.legend()
                plotter.save(fig, sub_folder=hists_result_name, name=t)
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.create_plot(figsize=(4.5, 8), nrows=2)
        plotter.plot_smooth(window=150, preplot=(fig, ax[0]), marker='',
                            xlabel='time (s)', ylabel='Information (bit)', title='')
        ax[0].axvline(0, ls='--', c='k')
        plotter.plot_smooth(window=150, preplot=(fig, ax[1]), title='', marker='', )
        ax[1].axvline(0, ls='--', c='k')
        ax[1].set_xlim((-2, 8))
        ax[1].set_ylim(ylim)
        plotter.save(fig)

    def __compute_entropy_evol(self, time_intervals, hists_result_name, info_result_name):
        column_names = self.exp.get_columns(info_result_name)
        for inter in column_names:
            for dt in time_intervals:
                hist = self.exp.get_df(hists_result_name).loc[pd.IndexSlice[dt, :], inter]
                entropy = get_entropy(hist)
                max_entropy = get_max_entropy(hist)

                self.exp.get_df(info_result_name).loc[dt, inter] = np.around(max_entropy - entropy, 3)

    def __plot_info_evol(self, info_result_name, ylabel, ylim_zoom):

        self.exp.remove_object(info_result_name)
        self.exp.load(info_result_name)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.create_plot(figsize=(4.5, 8), nrows=2)
        plotter.plot(preplot=(fig, ax[0]), xlabel='time (s)', ylabel=ylabel, title='')

        ax[0].axvline(0, ls='--', c='k')
        plotter.plot(preplot=(fig, ax[1]), title='')
        ax[1].axvline(0, ls='--', c='k')
        ax[1].set_xlim((-2, 8))
        ax[1].set_ylim(ylim_zoom)
        plotter.save(fig)

    def __plot_info_evol_vs_nbr_attach(self, info_result_name, xlabel, ylabel, ylim=None):

        self.exp.remove_object(info_result_name)
        self.exp.load(info_result_name)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.plot(xlabel=xlabel, ylabel=ylabel, title='')
        if ylim is not None:
            ax.set_ylim(ylim)
        plotter.save(fig)

    def __compute_information_around_attachments_evol(
            self, fct, dpi, start_frame_intervals, end_frame_intervals,
            variable_name, hists_result_name, info_result_name, init_frame_name,
            hists_label, hists_description, info_label, info_description, ylim_zoom, redo, redo_info):

        t0, t1, dt = -60, 60, 0.5
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)

        dtheta = np.pi * dpi
        bins = np.arange(0, np.pi + dtheta, dtheta)
        hists_index_values = np.around((bins[1:] + bins[:-1]) / 2., 3)

        column_names = [
            str([start_frame_intervals[i] / 100., end_frame_intervals[i] / 100.])
            for i in range(len(start_frame_intervals))]

        if redo:

            index_values = [(dt, theta) for dt in time_intervals for theta in hists_index_values]

            self.exp.add_new_empty_dataset(name=hists_result_name, index_names=['dt', 'food_direction_error'],
                                           column_names=column_names, index_values=index_values,
                                           category=self.category, label=hists_label, description=hists_description)
            self.exp.load(init_frame_name)
            self.change_first_frame(variable_name, init_frame_name)

            for t in time_intervals:
                for i in range(len(start_frame_intervals)):
                    index0 = start_frame_intervals[i]
                    index1 = end_frame_intervals[i]

                    df = self.exp.get_df(variable_name)[str(t)]
                    frames = df.index.get_level_values(id_frame_name)
                    index_location = (frames > index0) & (frames < index1)

                    df = df.loc[index_location]
                    hist = np.histogram(df.dropna(), bins=bins, normed=False)[0]
                    s = float(np.sum(hist))
                    hist = hist / s

                    self.exp.get_df(hists_result_name).loc[pd.IndexSlice[t, :], column_names[i]] = hist

            self.exp.remove_object(variable_name)
            self.exp.write(hists_result_name)

        else:
            self.exp.load(hists_result_name)

        if redo or redo_info:

            self.exp.add_new_empty_dataset(name=info_result_name, index_names='dt', column_names=column_names,
                                           index_values=time_intervals, category=self.category,
                                           label=info_label, description=info_description)

            fct(time_intervals, hists_result_name, info_result_name)

            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)

        ylabel = 'Information (bit)'

        self.__plot_info_evol(info_result_name, ylabel, ylim_zoom)

    def compute_information_mm1s_food_direction_error_around_outside_attachments_evol(
            self, redo=False, redo_info=False):

        variable_name = 'mm1s_food_direction_error_around_outside_attachments'
        self.exp.load(variable_name)
        init_frame_name = 'first_attachment_time_of_outside_ant'

        hists_result_name = 'histograms_' + variable_name+'_evol'
        info_result_name = 'information_' + variable_name+'_evol'

        hists_label = 'Histograms of the food direction error around outside attachments over time'
        hists_description = 'Time evolution of the histograms of the food direction error' \
                            ' for each time t in time_intervals, ' \
                            'which are times around outside ant attachments'

        info_label = 'Information of the food around outside attachments over time'
        info_description = 'Time evolution of the information of the food' \
                           ' (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside ant attachments'

        dx = 0.5
        start_frame_intervals = np.arange(0, 2.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        ylim_zoom = (0.2, 0.6)
        dpi = 1/12.
        self.__compute_information_around_attachments_evol(
            self.__compute_entropy_evol, dpi,  start_frame_intervals, end_frame_intervals,
            variable_name, hists_result_name, info_result_name, init_frame_name,
            hists_label, hists_description, info_label, info_description, ylim_zoom, redo, redo_info)

    def compute_information_mm1s_food_direction_error_around_non_outside_attachments_evol(
            self, redo=False, redo_info=False):

        variable_name = 'mm1s_food_direction_error_around_non_outside_attachments'
        self.exp.load(variable_name)
        init_frame_name = 'first_attachment_time_of_outside_ant'

        hists_result_name = 'histograms_' + variable_name+'_evol'
        info_result_name = 'information_' + variable_name+'_evol'

        hists_label = 'Histograms of the food direction error around outside attachments over time'
        hists_description = 'Time evolution of the histograms of the food direction error' \
                            ' for each time t in time_intervals, ' \
                            'which are times around non outside ant attachments'

        info_label = 'Information of the food around outside attachments over time'
        info_description = 'Time evolution of the information of the food' \
                           ' (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around non outside ant attachments'

        dx = 0.5
        start_frame_intervals = np.arange(-1, 2.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        ylim_zoom = (0.1, 0.6)
        dpi = 1/12.
        self.__compute_information_around_attachments_evol(
            self.__compute_entropy_evol, dpi,  start_frame_intervals, end_frame_intervals,
            variable_name, hists_result_name, info_result_name, init_frame_name,
            hists_label, hists_description, info_label, info_description, ylim_zoom, redo, redo_info)

    def compute_information_mm1s_food_direction_error_around_non_outside_attachments_evol_after_first_outside_attachment(
            self, redo=False, redo_info=False):

        variable_name = 'mm1s_food_direction_error_around_non_outside_attachments_after_first_outside_attachment'
        self.exp.load(variable_name)
        init_frame_name = 'first_attachment_time_of_outside_ant'

        hists_result_name = 'histograms_' + variable_name+'_evol'
        info_result_name = 'information_' + variable_name+'_evol'

        hists_label = 'Histograms of the food direction error around outside attachments over time'
        hists_description = 'Time evolution of the histograms of the food direction error' \
                            ' for each time t in time_intervals, ' \
                            'which are times around non outside ant attachments'

        info_label = 'Information of the food around outside attachments over time'
        info_description = 'Time evolution of the information of the food' \
                           ' (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around non outside ant attachments'

        dx = 0.5
        start_frame_intervals = np.arange(0, 2.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        ylim_zoom = (0.1, 0.7)
        dpi = 1/12.
        self.__compute_information_around_attachments_evol(
            self.__compute_entropy_evol, dpi,  start_frame_intervals, end_frame_intervals,
            variable_name, hists_result_name, info_result_name, init_frame_name,
            hists_label, hists_description, info_label, info_description, ylim_zoom, redo, redo_info)

    def compute_information_mm1s_food_direction_error_around_attachments_evol(
            self, redo=False, redo_info=False):

        variable_name = 'mm1s_food_direction_error_around_outside_attachments'
        variable_name2 = 'mm1s_food_direction_error_around_non_outside_attachments'
        self.exp.load([variable_name, variable_name2])
        self.exp.get_data_object(variable_name).df =\
            pd.concat([self.exp.get_df(variable_name), self.exp.get_df(variable_name2)])

        init_frame_name = 'first_attachment_time_of_outside_ant'

        hists_result_name = 'histograms_mm1s_food_direction_error_around_attachments_evol'
        info_result_name = 'information_mm1s_food_direction_error_around_attachments_evol'

        hists_label = 'Histograms of the food direction error around outside attachments over time'
        hists_description = 'Time evolution of the histograms of the food direction error' \
                            ' for each time t in time_intervals, ' \
                            'which are times around ant attachments'

        info_label = 'Information of the food around outside attachments over time'
        info_description = 'Time evolution of the information of the food' \
                           ' (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around ant attachments'

        dx = 0.5
        start_frame_intervals = np.arange(-1, 2.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        ylim_zoom = (0.1, 0.6)
        dpi = 1/12.
        self.__compute_information_around_attachments_evol(
            self.__compute_entropy_evol, dpi,  start_frame_intervals, end_frame_intervals,
            variable_name, hists_result_name, info_result_name, init_frame_name,
            hists_label, hists_description, info_label, info_description, ylim_zoom, redo, redo_info)

    def compute_information_mm1s_food_direction_error_over_nbr_new_attachments(
            self, redo=False, redo_info=False):

        attachment_name = 'carrying_intervals'

        fct_get_attachment = self._get_nb_attachments
        fct_info = self.__compute_entropy
        fct_info_evol = self.__compute_entropy_evol
        variable_name = 'mm1s_food_direction_error'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_mm1s_food_direction_error_over_nb_new_attachments'
        info_result_name = 'information_mm1s_food_direction_error_over_nb_new_attachments'

        hists_label = 'Histograms of the food direction error over the number of attachments'
        explanation = " over the number of attachments occurred the last 8 seconds. More precisely, to report an "\
                      " attachment influence, we count that an attachment is happening up to 8 seconds" \
                      " (or less if the attachment lasts less) after it occurs."
        hists_description = "Histograms of the food direction error"+explanation

        info_label = 'Information of the food over the number of attachments'
        info_description = "Information of the food (max entropy - entropy of the food direction error)"+explanation

        hists_result_evol_name = hists_result_name+'_evol'
        info_result_evol_name = info_result_name+'_evol'
        hists_label_evol = hists_label+' over time'
        hists_description_evol = 'Time evolution of '+hists_description
        info_label_evol = info_label+' over time'
        info_description_evol = 'Time evolution of '+info_description

        dx = 0.25
        start_frame_intervals = np.arange(-1, 2.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        dpi = 1/12.

        attach_intervals = range(20)

        self.__compute_information_over_nb_new_attachments(
            variable_name, attach_intervals, dpi, start_frame_intervals, end_frame_intervals,
            fct_info, fct_info_evol, fct_get_attachment, attachment_name,
            hists_description, hists_description_evol, hists_label, hists_label_evol, hists_result_name,
            hists_result_evol_name, info_description, info_description_evol, info_label, info_label_evol,
            info_result_name, info_result_evol_name, redo, redo_info)

        xlabel = 'Ratio of attachments'
        ylabel = 'Information (bit)'
        self.__plot_info_vs_nb_attach(info_result_name, xlabel, ylabel)
        self.__plot_info_evol_vs_nbr_attach(info_result_evol_name, xlabel, ylabel, ylim=(0, 3))

    def compute_information_mm1s_food_direction_error_over_nbr_new_outside_attachments(
            self, redo=False, redo_info=False):

        attachment_name = 'outside_ant_carrying_intervals'

        fct_get_attachment = self._get_nb_attachments
        fct_info = self.__compute_entropy
        fct_info_evol = self.__compute_entropy_evol
        variable_name = 'mm1s_food_direction_error'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_mm1s_food_direction_error_over_nb_new_outside_attachments'
        info_result_name = 'information_mm1s_food_direction_error_over_nb_new_outside_attachments'

        hists_label = 'Histograms of the food direction error over the number of outside attachments'
        explanation = " over the number of outside attachments occurred the last 8 seconds. More precisely," \
                      " to report an outside attachment influence, we count that an outside attachment is happening" \
                      " up to  8 seconds (or less if the attachment lasts less) after it occurs."
        hists_description = "Histograms of the food direction error" + explanation

        info_label = 'Information of the food over the number of outside attachments'
        info_description = "Information of the food (max entropy - entropy of the food direction error)" + explanation

        hists_result_evol_name = hists_result_name + '_evol'
        info_result_evol_name = info_result_name + '_evol'
        hists_label_evol = hists_label + ' over time'
        hists_description_evol = 'Time evolution of ' + hists_description
        info_label_evol = info_label + ' over time'
        info_description_evol = 'Time evolution of ' + info_description

        dx = 0.25
        start_frame_intervals = np.arange(-1, 2.5, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx * 60 * 100 * 2

        dpi = 1 / 12.

        attach_intervals = range(20)

        self.__compute_information_over_nb_new_attachments(
            variable_name, attach_intervals, dpi, start_frame_intervals, end_frame_intervals,
            fct_info, fct_info_evol, fct_get_attachment, attachment_name,
            hists_description, hists_description_evol, hists_label, hists_label_evol, hists_result_name,
            hists_result_evol_name, info_description, info_description_evol, info_label, info_label_evol,
            info_result_name, info_result_evol_name, redo, redo_info)

        xlabel = 'Ratio of attachments'
        ylabel = 'Information (bit)'
        self.__plot_info_vs_nb_attach(info_result_name, xlabel, ylabel)
        self.__plot_info_evol_vs_nbr_attach(info_result_evol_name, xlabel, ylabel)

    def compute_information_mm1s_food_direction_error_over_nbr_new_non_outside_attachments(
            self, redo=False, redo_info=False):

        attachment_name = 'non_outside_ant_carrying_intervals'

        fct_get_attachment = self._get_nb_attachments
        fct_info = self.__compute_entropy
        fct_info_evol = self.__compute_entropy_evol

        variable_name = 'mm1s_food_direction_error'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_mm1s_food_direction_error_over_nb_new_non_outside_attachments'
        info_result_name = 'information_mm1s_food_direction_error_over_nb_new_non_outside_attachments'

        hists_label = 'Histograms of the food direction error over the number of non outside attachments'
        explanation = " over the number of outside attachments occurred the last 8 seconds. More precisely," \
                      " to report an non outside attachment influence, we count that an non outside attachment" \
                      " is happening up to  8 seconds (or less if the attachment lasts less) after it occurs."
        hists_description = "Histograms of the food direction error" + explanation

        info_label = 'Information of the food over the number of non outside attachments'
        info_description = "Information of the food (max entropy - entropy of the food direction error)" + explanation

        hists_result_evol_name = hists_result_name + '_evol'
        info_result_evol_name = info_result_name + '_evol'
        hists_label_evol = hists_label + ' over time'
        hists_description_evol = 'Time evolution of ' + hists_description
        info_label_evol = info_label + ' over time'
        info_description_evol = 'Time evolution of ' + info_description

        dx = 0.25
        start_frame_intervals = np.arange(-1, 2.5, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx * 60 * 100 * 2

        dpi = 1 / 12.

        attach_intervals = range(20)

        xlabel = 'Number of attachments'
        ylabel = 'Information (bit)'
        self.__compute_information_over_nb_new_attachments(
            variable_name, attach_intervals, dpi, start_frame_intervals, end_frame_intervals,
            fct_info, fct_info_evol, fct_get_attachment, attachment_name,
            hists_description, hists_description_evol, hists_label, hists_label_evol, hists_result_name,
            hists_result_evol_name, info_description, info_description_evol, info_label, info_label_evol,
            info_result_name, info_result_evol_name, redo, redo_info)

        info_outside_name = 'information_mm1s_food_direction_error_over_nb_new_outside_attachments'
        info_name = 'information_mm1s_food_direction_error_over_nb_new_attachments'
        self.exp.load([info_outside_name, info_name])
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.plot(xlabel=xlabel, ylabel=ylabel, title='', label='Inside attachments')
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_outside_name))
        plotter.plot(preplot=(fig, ax), c='w', label='Outside attachments', title='', xlabel=xlabel)
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_name))
        plotter.plot(preplot=(fig, ax), c='r', label='All attachments', title='', xlabel=xlabel)
        ax.set_xlim((-0.5, 10))
        ax.legend()
        plotter.save(fig,
                     name='information_mm1s_food_direction_error_over_nb_new_outside_and_inside_attachments')

    def compute_information_mm1s_food_direction_error_over_ratio_new_attachments(
            self, redo=False, redo_info=False):

        attachment_name = 'outside_ant_carrying_intervals'

        fct_get_attachment = self._get_ratio_attachments
        fct_info = self.__compute_entropy
        fct_info_evol = self.__compute_entropy_evol

        variable_name = 'mm1s_food_direction_error'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_mm1s_food_direction_error_over_ratio_new_attachments'
        info_result_name = 'information_mm1s_food_direction_error_over_ratio_new_attachments'

        hists_label = 'Histograms of the food direction error over the ratio of outside attachments'
        explanation = " over the ratio of outside attachments occurred the last 8 seconds. More precisely," \
                      " to report an attachment influence, we count that an attachment" \
                      " is happening up to  8 seconds (or less if the attachment lasts less) after it occurs."
        hists_description = "Histograms of the food direction error" + explanation

        info_label = 'Information of the food over the ratio of outside attachments'
        info_description = "Information of the food (max entropy - entropy of the food direction error)" + explanation

        hists_result_evol_name = hists_result_name + '_evol'
        info_result_evol_name = info_result_name + '_evol'
        hists_label_evol = hists_label + ' over time'
        hists_description_evol = 'Time evolution of ' + hists_description
        info_label_evol = info_label + ' over time'
        info_description_evol = 'Time evolution of ' + info_description

        dx = 0.25
        start_frame_intervals = np.arange(-1, 2.5, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx * 60 * 100 * 2

        attach_intervals = np.around(np.arange(-0.2, 1.1, 0.2), 1)
        print(attach_intervals)

        dpi = 1 / 12.
        self.__compute_information_over_nb_new_attachments(
            variable_name, attach_intervals, dpi, start_frame_intervals, end_frame_intervals,
            fct_info, fct_info_evol, fct_get_attachment, attachment_name,
            hists_description, hists_description_evol, hists_label, hists_label_evol, hists_result_name,
            hists_result_evol_name, info_description, info_description_evol, info_label, info_label_evol,
            info_result_name, info_result_evol_name, redo, redo_info)

        xlabel = 'Ratio of attachments'
        ylabel = 'Information (bit)'
        self.__plot_info_vs_nb_attach(info_result_name, xlabel, ylabel)
        self.__plot_info_evol_vs_nbr_attach(info_result_evol_name, xlabel, ylabel)

    def compute_information_mm1s_food_direction_error_around_outside_manual_leading_attachments_evol(
            self, redo=False, redo_info=False):

        variable_name = 'mm1s_food_direction_error_around_outside_manual_leading_attachments'
        self.exp.load(variable_name)
        init_frame_name = 'first_attachment_time_of_outside_ant'

        hists_result_name = 'histograms_' + variable_name+'_evol'
        info_result_name = 'information_' + variable_name+'_evol'

        hists_label = 'Histograms of the food direction error around outside attachments over time'
        hists_description = 'Time evolution of the histograms of the food direction error' \
                            ' for each time t in time_intervals, ' \
                            'which are times around outside ant attachments'

        info_label = 'Information of the food around outside attachments over time'
        info_description = 'Time evolution of the information of the food' \
                           ' (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside ant attachments'

        dx = 0.75
        start_frame_intervals = np.arange(0, 2.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        ylim_zoom = (0.4, 3.)
        dpi = 1/12.
        self.__compute_information_around_attachments_evol(
            self.__compute_entropy_evol, dpi,  start_frame_intervals, end_frame_intervals,
            variable_name, hists_result_name, info_result_name, init_frame_name,
            hists_label, hists_description, info_label, info_description, ylim_zoom, redo, redo_info)

    def compute_information_mm1s_food_direction_error_around_inside_manual_leading_attachments_evol(
            self, redo=False, redo_info=False):

        variable_name = 'mm1s_food_direction_error_around_inside_manual_leading_attachments'
        self.exp.load(variable_name)
        init_frame_name = 'first_attachment_time_of_outside_ant'

        hists_result_name = 'histograms_' + variable_name+'_evol'
        info_result_name = 'information_' + variable_name+'_evol'

        hists_label = 'Histograms of the food direction error around inside attachments over time'
        hists_description = 'Time evolution of the histograms of the food direction error' \
                            ' for each time t in time_intervals, ' \
                            'which are times around outside ant attachments'

        info_label = 'Information of the food around inside attachments over time'
        info_description = 'Time evolution of the information of the food' \
                           ' (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside ant attachments'

        dx = 0.75
        start_frame_intervals = np.arange(-2, 2.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        ylim_zoom = (0.4, 3.)
        dpi = 1/12.
        self.__compute_information_around_attachments_evol(
            self.__compute_entropy_evol, dpi,  start_frame_intervals, end_frame_intervals,
            variable_name, hists_result_name, info_result_name, init_frame_name,
            hists_label, hists_description, info_label, info_description, ylim_zoom, redo, redo_info)

    def compute_information_mm1s_food_direction_error_around_outside_manual_following_attachments_evol(
            self, redo=False, redo_info=False):

        variable_name = 'mm1s_food_direction_error_around_outside_manual_following_attachments'
        self.exp.load(variable_name)
        init_frame_name = 'first_attachment_time_of_outside_ant'

        hists_result_name = 'histograms_' + variable_name+'_evol'
        info_result_name = 'information_' + variable_name+'_evol'

        hists_label = 'Histograms of the food direction error around outside attachments over time'
        hists_description = 'Time evolution of the histograms of the food direction error' \
                            ' for each time t in time_intervals, ' \
                            'which are times around outside ant attachments'

        info_label = 'Information of the food around outside attachments over time'
        info_description = 'Time evolution of the information of the food' \
                           ' (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside ant attachments'

        dx = 0.75
        start_frame_intervals = np.arange(0, 2.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        ylim_zoom = (0.4, 3.)
        dpi = 1/12.
        self.__compute_information_around_attachments_evol(
            self.__compute_entropy_evol, dpi,  start_frame_intervals, end_frame_intervals,
            variable_name, hists_result_name, info_result_name, init_frame_name,
            hists_label, hists_description, info_label, info_description, ylim_zoom, redo, redo_info)

    def compute_information_mm1s_food_direction_error_around_inside_manual_following_attachments_evol(
            self, redo=False, redo_info=False):

        variable_name = 'mm1s_food_direction_error_around_inside_manual_following_attachments'
        self.exp.load(variable_name)
        init_frame_name = 'first_attachment_time_of_outside_ant'

        hists_result_name = 'histograms_' + variable_name+'_evol'
        info_result_name = 'information_' + variable_name+'_evol'

        hists_label = 'Histograms of the food direction error around outside attachments over time'
        hists_description = 'Time evolution of the histograms of the food direction error' \
                            ' for each time t in time_intervals, ' \
                            'which are times around outside ant attachments'

        info_label = 'Information of the food around outside attachments over time'
        info_description = 'Time evolution of the information of the food' \
                           ' (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside ant attachments'

        dx = 0.75
        start_frame_intervals = np.arange(-2, 2.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        ylim_zoom = (0., 2.)
        dpi = 1/12.
        self.__compute_information_around_attachments_evol(
            self.__compute_entropy_evol, dpi,  start_frame_intervals, end_frame_intervals,
            variable_name, hists_result_name, info_result_name, init_frame_name,
            hists_label, hists_description, info_label, info_description, ylim_zoom, redo, redo_info)

    def __compute_information_over_nb_new_attachments(
            self, variable_name, attach_intervals, dpi, start_frame_intervals, end_frame_intervals,
            fct_info, fct_info_evol, fct_get_attachment, attachment_name,
            hists_description, hists_description_evol, hists_label, hists_label_evol, hists_result_name,
            hists_result_evol_name, info_description, info_description_evol, info_label, info_label_evol,
            info_result_name, info_result_evol_name, redo, redo_info):

        dtheta = np.pi * dpi
        bins = np.arange(0, np.pi + dtheta, dtheta)
        hists_index_values = np.around((bins[1:] + bins[:-1]) / 2., 3)
        column_names = [
            str([start_frame_intervals[i] / 100., end_frame_intervals[i] / 100.])
            for i in range(len(start_frame_intervals))]

        self._get_hist_evol(variable_name, attachment_name, column_names, hists_result_name, hists_result_evol_name,
                            attach_intervals, bins, start_frame_intervals, end_frame_intervals, fct_get_attachment,
                            hists_label, hists_description, hists_label_evol, hists_description_evol,
                            hists_index_values, redo)

        if redo or redo_info:

            self.exp.add_new_empty_dataset(name=info_result_name, index_names='time', column_names='info',
                                           index_values=attach_intervals, category=self.category,
                                           label=info_label, description=info_description)
            fct_info(attach_intervals, hists_result_name, info_result_name)
            self.exp.write(info_result_name)

            self.exp.add_new_empty_dataset(name=info_result_evol_name, index_names='nb_attach',
                                           column_names=column_names, index_values=attach_intervals,
                                           category=self.category,
                                           label=info_label_evol, description=info_description_evol)

            fct_info_evol(attach_intervals, hists_result_evol_name, info_result_evol_name)

            self.exp.write(info_result_evol_name)

        else:
            self.exp.load(info_result_name)
            self.exp.load(info_result_evol_name)

    def _get_hist_evol(self, variable_name, attachment_name, column_names, hists_result_name, hists_result_evol_name,
                       attach_intervals, bins, start_frame_intervals, end_frame_intervals, fct_get_attachment,
                       hists_label, hists_description, hists_label_evol, hists_description_evol, hists_index_values,
                       redo):
        if redo:
            init_frame_name = 'first_attachment_time_of_outside_ant'
            self.change_first_frame(variable_name, init_frame_name)

            fct_get_attachment(attachment_name, variable_name, attach_intervals)

            print('create res obj')
            index_values = [(nb_attach, theta) for nb_attach in attach_intervals for theta in hists_index_values]
            self.exp.add_new_empty_dataset(name=hists_result_evol_name,
                                           index_names=['nb_attachments', 'food_direction_error'],
                                           column_names=column_names, index_values=index_values,
                                           category=self.category, label=hists_label_evol,
                                           description=hists_description_evol)
            print('create res obj')
            self.exp.add_new_empty_dataset(
                name=hists_result_name, index_names='nb_attachments', column_names=attach_intervals,
                index_values=hists_index_values,
                category=self.category, label=hists_label, description=hists_description)

            print('create res obj')
            nb_attach_list = set(self.exp.get_index(variable_name).get_level_values('nb_attachments'))
            print(nb_attach_list)
            for nb in nb_attach_list:
                print(nb)
                df = self.exp.get_df(variable_name).loc[pd.IndexSlice[:, :, nb], :].dropna()
                if len(df) > 100 and len(set(df.index.get_level_values(id_exp_name))) > 2:
                    hist = np.histogram(df.dropna(), bins=bins, normed=False)[0]
                    s = float(np.sum(hist))
                    hist = hist / s

                    self.exp.get_df(hists_result_name)[nb] = hist

                for i in range(len(start_frame_intervals)):
                    frame0 = start_frame_intervals[i]
                    frame1 = end_frame_intervals[i]

                    df = self.exp.get_df(variable_name).loc[pd.IndexSlice[:, frame0:frame1, nb], :].dropna()
                    if len(df) > 100 and len(set(df.index.get_level_values(id_exp_name))) > 2:
                        hist = np.histogram(df.dropna(), bins=bins, normed=False)[0]
                        s = float(np.sum(hist))
                        hist = hist / s

                        self.exp.get_df(hists_result_evol_name).loc[pd.IndexSlice[nb, :], column_names[i]] = hist

            self.exp.remove_object(variable_name)
            self.exp.write(hists_result_name)

        else:
            self.exp.load(hists_result_name)

    def _get_nb_attachments(self, attachment_name, variable_name, attach_intervals):
        if attach_intervals:
            pass

        idx = self.exp.get_index(variable_name)
        nb_attachment_name = 'nb_attachment'
        self.exp.add_new_empty_dataset(
            name=nb_attachment_name, index_names=[id_exp_name, id_frame_name], column_names=nb_attachment_name,
            index_values=idx, fill_value=0, replace=True)

        self.exp.load([attachment_name, 'fps'])
        attachment_arr = self.exp.get_df(attachment_name).reset_index().values
        for id_exp, id_ant, frame0, time in attachment_arr:
            fps = self.exp.get_value('fps', id_exp)
            dframe = time * fps
            frame1 = int(frame0 + min(dframe, 8 * fps))
            self.exp.get_df(nb_attachment_name).loc[pd.IndexSlice[int(id_exp), int(frame0):frame1], :] += 1

        idx2 = list(zip(self.exp.get_index(nb_attachment_name).get_level_values(id_exp_name),
                        self.exp.get_index(nb_attachment_name).get_level_values(id_frame_name),
                        self.exp.get_df(nb_attachment_name).values.ravel()))
        self.exp.get_df(variable_name).index = pd.MultiIndex.from_tuples(
            idx2, names=[id_exp_name, id_frame_name, 'nb_attachments'])

    def _get_ratio_attachments(self, outside_attachment_name, variable_name, attach_intervals):

        inside_attachment_name = 'in' + outside_attachment_name[3:]
        self.exp.load([outside_attachment_name, inside_attachment_name, 'fps'])
        idx = self.exp.get_index(variable_name)

        for i, (nb_attachment_name, attachment_name2) in enumerate(
                [('nb_outside_attachment', outside_attachment_name),
                 ('nb_inside_attachment', inside_attachment_name)]):
            print(nb_attachment_name)

            self.exp.add_new_empty_dataset(
                name=nb_attachment_name, index_names=[id_exp_name, id_frame_name],
                column_names='nb', index_values=idx, fill_value=0, replace=True)

            attachment_arr = self.exp.get_df(attachment_name2).reset_index().values
            for id_exp, id_ant, frame0, time in attachment_arr:
                fps = self.exp.get_value('fps', id_exp)
                dframe = time * fps
                frame1 = int(frame0 + min(dframe, 8 * fps))
                self.exp.get_df(nb_attachment_name).loc[pd.IndexSlice[int(id_exp), int(frame0):frame1], :] += 1

        nb_outside_attach_arr = self.exp.get_df('nb_outside_attachment').values.ravel()
        nb_inside_attach_arr = self.exp.get_df('nb_inside_attachment').values.ravel()
        idx_ratio = nb_outside_attach_arr/(nb_inside_attach_arr+nb_outside_attach_arr).astype(float)
        idx_ratio[np.isnan(idx_ratio)] = -0.1
        for i in range(len(idx_ratio)):
            idx_ratio[i] = get_interval_containing(idx_ratio[i], attach_intervals)

        idx2 = list(zip(self.exp.get_index('nb_outside_attachment').get_level_values(id_exp_name),
                        self.exp.get_index('nb_outside_attachment').get_level_values(id_frame_name),
                        idx_ratio))
        self.exp.get_df(variable_name).index = pd.MultiIndex.from_tuples(
            idx2, names=[id_exp_name, id_frame_name, 'nb_attachments'])

    def compute_fisher_information_mm1s_food_direction_error_around_first_outside_attachments(self, redo=False):

        variable_name = 'mm1s_food_direction_error_around_outside_attachments'

        info_result_name = 'fisher information_mm1s_food_direction_error_around_first_outside_attachments'

        info_label = 'Fisher information of the food around outside first outside attachments'
        info_description = 'Fisher information of the food (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around first outside ant attachments'

        ylim = (0.1, 0.2)
        self.__compute_fisher_information_around_outside_attachments(variable_name, info_result_name,
                                                                     info_label, info_description, ylim, redo)

    def __compute_fisher_information_around_outside_attachments(self, variable_name, info_result_name, info_label,
                                                                info_description, ylim, redo):
        self.exp.load([variable_name])
        rank_list = np.arange(1., 11.)
        t0, t1, dt = -60, 60, 0.1
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)
        index_values = [(th, t) for th in rank_list for t in time_intervals]

        if redo:

            self.exp.add_new_empty_dataset(name=info_result_name, index_names='time', column_names=rank_list,
                                           index_values=time_intervals, category=self.category,
                                           label=info_label, description=info_description)

            first_attach_name = 'first_attachments'
            self.__extract_10first_attachments(first_attach_name, variable_name)

            for (th, t) in index_values:
                values = self.exp.get_df(first_attach_name).loc[pd.IndexSlice[:, th], str(t)].dropna()
                fisher_info = compute_fisher_information_uniform_von_mises(values)
                # fisher_info = 1/np.var(values)

                self.exp.get_df(info_result_name).loc[t, th] = np.around(fisher_info, 5)

            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.create_plot(figsize=(4.5, 8), nrows=2)
        plotter.plot_smooth(window=150, preplot=(fig, ax[0]), marker='',
                            xlabel='time (s)', ylabel='Fisher information', title='')
        ax[0].axvline(0, ls='--', c='k')
        plotter.plot_smooth(window=150, preplot=(fig, ax[1]), title='', marker='', )
        ax[1].axvline(0, ls='--', c='k')
        ax[1].set_xlim((-2, 8))
        ax[1].set_ylim(ylim)
        plotter.save(fig)

    def __compute_entropy_around_outside_attachments(self, hists_result_name, info_result_name):
        for (th, t) in self.exp.get_index(hists_result_name):
            hist = self.exp.get_df(hists_result_name).loc[(th, t)]
            entropy = get_entropy(hist)
            max_entropy = get_max_entropy(hist)

            self.exp.get_df(info_result_name).loc[t, th] = np.around(max_entropy - entropy, 6)

    def __extract_10first_attachments(self, first_attach_name, variable_name):
        first_attach_index = [(id_exp, th) for id_exp in self.exp.id_exp_list for th in range(1, 11)]
        self.exp.add_new_empty_dataset(name=first_attach_name, index_names=[id_exp_name, 'rank'],
                                       column_names=self.exp.get_df(variable_name).columns,
                                       index_values=first_attach_index, replace=True)

        def get_10first_attachment(df):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            print(id_exp)
            arr = np.array(df.iloc[:10, :])
            self.exp.get_df(first_attach_name).loc[pd.IndexSlice[id_exp, :len(arr)], :] = arr

        self.exp.groupby(variable_name, id_exp_name, get_10first_attachment)

    def compute_information_mm1s_food_direction_error_around_the_first_outside_attachment(self, redo=False):

        variable_name = 'mm1s_food_direction_error'

        info_result_name = 'information_mm1s_food_direction_error_around_the_first_outside_attachment'

        info_label = 'Information of the food around outside the first outside attachment'
        info_description = 'Information of the food (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around the first outside ant attachment'

        self.__compute_information_around_first_attachment(self.__compute_entropy_around_first_attachment,
                                                           variable_name, info_result_name, info_label,
                                                           info_description, redo)

    def __compute_information_around_first_attachment(self, fct, variable_name, info_result_name, info_label,
                                                      info_description, redo):

        first_attachment_name = 'first_attachment_time_of_outside_ant'
        self.exp.load([variable_name, first_attachment_name, 'fps'])

        t0, t1, dt = -100, 400, 0.5
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)
        t0, t1, dt = -100, 400, 5
        time_intervals_to_plot = np.around(np.arange(t0, t1 + dt, dt), 1)
        dtheta = np.pi / 12.
        bins = np.arange(0, np.pi + dtheta, dtheta)
        bins2 = np.around((bins[1:] + bins[:-1]) / 2., 3)

        if redo:
            self.exp.add_new_empty_dataset(name='temp', index_names='time', column_names=self.exp.id_exp_list,
                                           index_values=time_intervals, replace=True)

            def get_variable_around_first_attachment(df1: pd.DataFrame):
                id_exp = df1.index.get_level_values(id_exp_name)[0]
                fps = self.exp.get_value('fps', id_exp)
                first_attach = self.exp.get_value(first_attachment_name, id_exp)

                df2 = df1.loc[id_exp, :]
                df2.index -= first_attach
                df2.index /= fps
                df2 = df2.reindex(time_intervals)

                self.exp.get_df('temp')[id_exp] = df2[variable_name]

            self.exp.groupby(variable_name, id_exp_name, get_variable_around_first_attachment)

            self.exp.add_new_empty_dataset(name=info_result_name, index_names='time', column_names='information',
                                           index_values=time_intervals, category=self.category,
                                           label=info_label, description=info_description)

            fct(bins, bins2, info_result_name, time_intervals, time_intervals_to_plot)

            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.create_plot(figsize=(4.5, 8), nrows=2)
        plotter.plot(preplot=(fig, ax[0]), xlabel='time (s)', ylabel='Information (bit)', title='')
        ax[0].axvline(0, ls='--', c='k')
        plotter.plot(preplot=(fig, ax[1]), title='')
        ax[1].axvline(0, ls='--', c='k')
        ax[1].set_xlim((-2, 8))
        ax[1].set_ylim((0, 1))
        plotter.save(fig)

    def __compute_entropy_around_first_attachment(self, bins, bins2, info_result_name, time_intervals,
                                                  time_intervals_to_plot):
        max_entropy = get_max_entropy(bins2)
        for t in time_intervals:
            values = self.exp.get_df('temp').loc[t].dropna()
            hist = np.histogram(values, bins=bins, density=False)[0]
            s = float(np.sum(hist))
            hist = hist / s
            entropy = get_entropy(hist)

            self.exp.get_df(info_result_name).loc[t] = np.around(max_entropy - entropy, 2)

            if not (np.isnan(entropy)) and t in time_intervals_to_plot:
                df = pd.DataFrame(data=hist, index=bins2, columns=['hist'])
                self.exp.add_new_dataset_from_df(df=df, name='hist', category=self.category, replace=True)

                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('hist'))
                fig, ax = plotter.plot(xlabel='Food direction error', ylabel='Probability')

                plotter.save(fig, sub_folder=info_result_name, name=t)

    def compute_fisher_information_mm1s_food_direction_error_around_the_first_outside_attachment(self, redo=False):

        variable_name = 'mm1s_food_direction_error'

        info_result_name = 'fisher_information_mm1s_food_direction_error_around_the_first_outside_attachment'

        info_label = 'Fisher information of the food around outside the first outside attachment'
        info_description = 'Fisher information of the food (1/(3variable))' \
                           ' for each time t in time_intervals, which are times around the first outside ant attachment'

        self.__compute_fisher_information_around_first_attachment(variable_name, info_result_name, info_label,
                                                                  info_description, redo)

    def __compute_fisher_information_around_first_attachment(self, variable_name, info_result_name, info_label,
                                                             info_description, redo):

        first_attachment_name = 'first_attachment_time_of_outside_ant'
        self.exp.load([variable_name, first_attachment_name, 'fps'])

        t0, t1, dt = -100, 400, 0.5
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)

        if redo:
            self.exp.add_new_empty_dataset(name='temp', index_names='time', column_names=self.exp.id_exp_list,
                                           index_values=time_intervals, replace=True)

            def get_variable_around_first_attachment(df1: pd.DataFrame):
                id_exp = df1.index.get_level_values(id_exp_name)[0]
                fps = self.exp.get_value('fps', id_exp)
                first_attach = self.exp.get_value(first_attachment_name, id_exp)

                df2 = df1.loc[id_exp, :]
                df2.index -= first_attach
                df2.index /= fps
                df2 = df2.reindex(time_intervals)

                self.exp.get_df('temp')[id_exp] = df2[variable_name]

            self.exp.groupby(variable_name, id_exp_name, get_variable_around_first_attachment)

            self.exp.add_new_empty_dataset(name=info_result_name, index_names='time', column_names='information',
                                           index_values=time_intervals, category=self.category,
                                           label=info_label, description=info_description)

            for t in time_intervals:
                values = self.exp.get_df('temp').loc[t].dropna()
                fisher_info = compute_fisher_information_uniform_von_mises(values)
                # fisher_info = 1/np.var(values)

                self.exp.get_df(info_result_name).loc[t] = np.around(fisher_info, 5)

            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.create_plot(figsize=(4.5, 8), nrows=2)
        plotter.plot(preplot=(fig, ax[0]), xlabel='time (s)', ylabel='Information (bit)', title='')
        ax[0].axvline(0, ls='--', c='k')
        ax[0].set_ylim((0, 1))

        plotter.plot(preplot=(fig, ax[1]), title='')
        ax[1].axvline(0, ls='--', c='k')
        ax[1].set_xlim((-2, 8))
        ax[1].set_ylim((0, 0.2))
        plotter.save(fig)

    def compute_fisher_information_mm1s_food_direction_error_around_attachments(self, redo=False):

        variable_name = 'mm1s_food_direction_error_around_outside_attachments'
        variable_name2 = 'mm1s_food_direction_error_around_non_outside_attachments'
        self.exp.load([variable_name, variable_name2])
        self.exp.get_data_object(variable_name).df =\
            pd.concat([self.exp.get_df(variable_name), self.exp.get_df(variable_name2)])

        info_result_name = 'fisher_information_mm1s_food_direction_error_around_attachments'

        info_label = 'Fisher information of the food around outside attachments'
        info_description = 'Fisher information of the food (1/(3variance))' \
                           ' for each time t in time_intervals, which are times around ant attachments'

        ylim_zoom = (0.35, 0.45)
        self.__compute_fisher_information_around_attachments(variable_name, info_result_name,
                                                             info_label, info_description, ylim_zoom, redo)

    def compute_fisher_information_mm1s_food_direction_error_around_outside_attachments(self, redo=False):

        variable_name = 'mm1s_food_direction_error_around_outside_attachments'
        self.exp.load(variable_name)

        info_result_name = 'fisher_information_' + variable_name

        info_label = 'Fisher information of the food around outside attachments'
        info_description = 'Fisher information of the food (1/(3variance))' \
                           ' for each time t in time_intervals, which are times around outside ant attachments'

        ylim_zoom = (0.4, 0.6)
        self.__compute_fisher_information_around_attachments(variable_name, info_result_name,
                                                             info_label, info_description, ylim_zoom, redo)

    def __compute_fisher_information_around_attachments(self, variable_name, info_result_name,
                                                        info_label, info_description, ylim_zoom, redo):
        t0, t1, dt = -60, 60, 0.5
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)
        if redo:

            self.exp.add_new_empty_dataset(name=info_result_name, index_names='time', column_names='info',
                                           index_values=time_intervals, category=self.category,
                                           label=info_label, description=info_description)

            self._compute_fisher_info(variable_name, info_result_name, time_intervals)

            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)

        ylabel = 'Fisher information'

        self.__plot_info('', info_result_name, time_intervals, ylabel, ylim=None, ylim_zoom=ylim_zoom,
                         redo=False, redo_plot_hist=False)

    def _compute_fisher_info(self, variable_name, info_result_name, time_intervals):
        for t in time_intervals:
            values = self.exp.get_df(variable_name)[str(t)].dropna()
            fisher_info = compute_fisher_information_uniform_von_mises(values)
            # fisher_info = 1 / np.var(values)

            self.exp.get_df(info_result_name).loc[t] = np.around(fisher_info, 5)

    def compute_fisher_information_mm1s_food_direction_error_around_isolated_outside_attachments(self, redo=False):

        variable_name = 'mm1s_food_direction_error_around_isolated_outside_attachments'
        self.exp.load(variable_name)

        info_result_name = 'fisher_information_' + variable_name

        info_label = 'Fisher information of the food around isolated outside attachments'
        info_description = 'Fisher information of the food (1/(3variance))' \
                           ' for each time t in time_intervals, which are times around isolated outside ant attachments'

        ylim_zoom = (0.35, 0.50)
        self.__compute_fisher_information_around_attachments(variable_name, info_result_name,
                                                             info_label, info_description, ylim_zoom, redo)

    def compute_fisher_information_mm1s_food_direction_error_around_isolated_non_outside_attachments(self, redo=False):

        variable_name = 'mm1s_food_direction_error_around_isolated_non_outside_attachments'
        self.exp.load(variable_name)

        info_result_name = 'fisher_information_' + variable_name

        info_label = 'Fisher information of the food around isolated non outside attachments'
        info_description = 'Fisher information of the food (1/(3variance))' \
                           ' for each time t in time_intervals,' \
                           ' which are times around isolated non outside ant attachments'

        ylim_zoom = (0.35, 0.6)
        self.__compute_fisher_information_around_attachments(variable_name, info_result_name,
                                                             info_label, info_description, ylim_zoom, redo)

    def compute_fisher_information_mm1s_food_direction_error_around_non_outside_attachments(self, redo=False):

        variable_name = 'mm1s_food_direction_error_around_non_outside_attachments'
        self.exp.load(variable_name)

        info_result_name = 'fisher_information_' + variable_name

        info_label = 'Fisher information of the food around non outside attachments'
        info_description = 'Fisher information of the food (1/(3variance))' \
                           ' for each time t in time_intervals, which are times around non outside ant attachments'

        ylim_zoom = (0.3, 0.4)
        self.__compute_fisher_information_around_attachments(variable_name, info_result_name,
                                                             info_label, info_description, ylim_zoom, redo)

    def compute_fisher_information_mm1s_food_direction_error_over_nbr_new_attachments(
            self, redo=False):

        attachment_name = 'carrying_intervals'

        fct_get_attachment = self._get_nb_attachments
        fct_info = self._compute_fisher_info_nb_attach
        fct_info_evol = self.__compute_fisher_info_evol
        variable_name = 'mm1s_food_direction_error'
        self.exp.load(variable_name)

        info_result_name = 'fisher_information_mm1s_food_direction_error_over_nb_new_attachments'

        explanation = " over the number of attachments occurred the last 8 seconds. More precisely, to report an "\
                      " attachment influence, we count that an attachment is happening up to 8 seconds" \
                      " (or less if the attachment lasts less) after it occurs."

        info_label = 'Fisher information of the food over the number of attachments'
        info_description = "Fisher information of the food (1/variance of the food direction error)"+explanation

        info_result_evol_name = info_result_name+'_evol'
        info_label_evol = info_label+' over time'
        info_description_evol = 'Time evolution of '+info_description

        dx = 0.25
        start_frame_intervals = np.arange(-1, 2.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        attach_intervals = range(20)

        self.__compute_fisher_information_over_nb_new_attachments(
            variable_name, attach_intervals, start_frame_intervals, end_frame_intervals,
            fct_info, fct_info_evol, fct_get_attachment, attachment_name,
            info_description, info_description_evol, info_label, info_label_evol,
            info_result_name, info_result_evol_name, redo)

        xlabel = 'Number of attachments'
        ylabel = 'Fisher information'
        self.__plot_info_vs_nb_attach(info_result_name, xlabel, ylabel)
        self.__plot_info_evol_vs_nbr_attach(info_result_evol_name, xlabel, ylabel, ylim=(0, 3))

    def compute_fisher_information_mm1s_food_direction_error_over_nbr_new_outside_attachments(
            self, redo=False):

        attachment_name = 'outside_carrying_intervals'

        fct_get_attachment = self._get_nb_attachments
        fct_info = self._compute_fisher_info_nb_attach
        fct_info_evol = self.__compute_fisher_info_evol
        variable_name = 'mm1s_food_direction_error'
        self.exp.load(variable_name)

        info_result_name = 'fisher_information_mm1s_food_direction_error_over_nb_new_outside_attachments'

        explanation = " over the number of outside attachments occurred the last 8 seconds. More precisely, to report" \
                      " an  outside attachment influence, we count that an attachment is happening up to 8 seconds" \
                      " (or less if the outside attachment lasts less) after it occurs."

        info_label = 'Fisher information of the food over the number of outside attachments'
        info_description = "Fisher information of the food (1/variance of the food direction error)"+explanation

        info_result_evol_name = info_result_name+'_evol'
        info_label_evol = info_label+' over time'
        info_description_evol = 'Time evolution of '+info_description

        dx = 0.25
        start_frame_intervals = np.arange(-1, 2.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        attach_intervals = range(20)

        self.__compute_fisher_information_over_nb_new_attachments(
            variable_name, attach_intervals, start_frame_intervals, end_frame_intervals,
            fct_info, fct_info_evol, fct_get_attachment, attachment_name,
            info_description, info_description_evol, info_label, info_label_evol,
            info_result_name, info_result_evol_name, redo)

        xlabel = 'Number of attachments'
        ylabel = 'Fisher information'
        self.__plot_info_vs_nb_attach(info_result_name, xlabel, ylabel, ylim=(0, 2))
        self.__plot_info_evol_vs_nbr_attach(info_result_evol_name, xlabel, ylabel, ylim=(0, 2))

    def compute_fisher_information_mm1s_food_direction_error_over_nbr_new_non_outside_attachments(
            self, redo=False):

        attachment_name = 'inside_carrying_intervals'

        fct_get_attachment = self._get_nb_attachments
        fct_info = self._compute_fisher_info_nb_attach
        fct_info_evol = self.__compute_fisher_info_evol
        variable_name = 'mm1s_food_direction_error'
        self.exp.load(variable_name)

        info_result_name = 'fisher_information_mm1s_food_direction_error_over_nb_new_non_outside_attachments'

        explanation = " over the number of non outside attachments occurred the last 8 seconds. More precisely," \
                      " to report an  outside non attachment influence, we count that an attachment is happening up" \
                      " to 8 seconds (or less if the outside attachment lasts less) after it occurs."

        info_label = 'Fisher information of the food over the number of non outside attachments'
        info_description = "Fisher information of the food (1/variance of the food direction error)"+explanation

        info_result_evol_name = info_result_name+'_evol'
        info_label_evol = info_label+' over time'
        info_description_evol = 'Time evolution of '+info_description

        dx = 0.25
        start_frame_intervals = np.arange(-1, 2.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        attach_intervals = range(20)

        self.__compute_fisher_information_over_nb_new_attachments(
            variable_name, attach_intervals, start_frame_intervals, end_frame_intervals,
            fct_info, fct_info_evol, fct_get_attachment, attachment_name,
            info_description, info_description_evol, info_label, info_label_evol,
            info_result_name, info_result_evol_name, redo)

        xlabel = 'Number of attachments'
        ylabel = 'Fisher information'
        self.__plot_info_vs_nb_attach(info_result_name, xlabel, ylabel, ylim=(0, 2))
        self.__plot_info_evol_vs_nbr_attach(info_result_evol_name, xlabel, ylabel)

        info_outside_name = 'fisher_information_mm1s_food_direction_error_over_nb_new_outside_attachments'
        info_name = 'fisher_information_mm1s_food_direction_error_over_nb_new_attachments'
        self.exp.load([info_outside_name, info_name])
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.plot(xlabel=xlabel, ylabel=ylabel, title='', label='Inside attachments')
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_outside_name))
        plotter.plot(preplot=(fig, ax), c='w', label='Outside attachments', title='', xlabel=xlabel)
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_name))
        plotter.plot(preplot=(fig, ax), c='r', label='All attachments', title='', xlabel=xlabel)
        ax.set_xlim((-0.5, 10))
        ax.legend()
        plotter.save(fig,
                     name='fisher_information_mm1s_food_direction_error_over_nb_new_outside_and_inside_attachments')

    def compute_fisher_information_mm1s_food_direction_error_over_ratio_new_attachments(
            self, redo=False):

        attachment_name = 'outside_carrying_intervals'

        fct_get_attachment = self._get_ratio_attachments
        fct_info = self._compute_fisher_info_nb_attach
        fct_info_evol = self.__compute_fisher_info_evol
        variable_name = 'mm1s_food_direction_error'
        self.exp.load(variable_name)

        info_result_name = 'fisher_information_mm1s_food_direction_error_over_ratio_new_attachments'

        explanation = " over the ratio of outside attachments occurred the last 8 seconds. More precisely," \
                      " to report an attachment influence, we count that an attachment is happening up" \
                      " to 8 seconds (or less if the outside attachment lasts less) after it occurs."

        info_label = 'Fisher information of the food over the ratio of attachments'
        info_description = "Fisher information of the food (1/variance of the food direction error)"+explanation

        info_result_evol_name = info_result_name+'_evol'
        info_label_evol = info_label+' over time'
        info_description_evol = 'Time evolution of '+info_description

        dx = 0.25
        start_frame_intervals = np.arange(-1, 2.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        attach_intervals = np.around(np.arange(-0.2, 1.1, 0.2), 1)

        self.__compute_fisher_information_over_nb_new_attachments(
            variable_name, attach_intervals, start_frame_intervals, end_frame_intervals,
            fct_info, fct_info_evol, fct_get_attachment, attachment_name,
            info_description, info_description_evol, info_label, info_label_evol,
            info_result_name, info_result_evol_name, redo)

        xlabel = 'Number of attachments'
        ylabel = 'Fisher information'
        self.__plot_info_vs_nb_attach(info_result_name, xlabel, ylabel)
        self.__plot_info_evol_vs_nbr_attach(info_result_evol_name, xlabel, ylabel, ylim=(0, 3))

    def __compute_fisher_information_over_nb_new_attachments(
            self, variable_name, attach_intervals, start_frame_intervals, end_frame_intervals,
            fct_info, fct_info_evol, fct_get_attachment, attachment_name,
            info_description, info_description_evol, info_label, info_label_evol,
            info_result_name, info_result_evol_name, redo):

        column_names = [
            str([start_frame_intervals[i] / 100., end_frame_intervals[i] / 100.])
            for i in range(len(start_frame_intervals))]

        if redo:
            init_frame_name = 'first_attachment_time_of_outside_ant'
            self.change_first_frame(variable_name, init_frame_name)

            fct_get_attachment(attachment_name, variable_name, attach_intervals)

            self.exp.add_new_empty_dataset(name=info_result_name, index_names='time', column_names='info',
                                           index_values=attach_intervals, category=self.category,
                                           label=info_label, description=info_description)
            fct_info(variable_name, info_result_name)
            self.exp.write(info_result_name)

            self.exp.add_new_empty_dataset(name=info_result_evol_name, index_names='nb_attach',
                                           column_names=column_names, index_values=attach_intervals,
                                           category=self.category,
                                           label=info_label_evol, description=info_description_evol)

            fct_info_evol(start_frame_intervals, end_frame_intervals, variable_name, info_result_evol_name)

            self.exp.write(info_result_evol_name)

        else:
            self.exp.load(info_result_name)
            self.exp.load(info_result_evol_name)

    def _compute_fisher_info_nb_attach(self, variable_name, info_result_name):
        nb_intervals = set(self.exp.get_index(variable_name).get_level_values('nb_attachments'))
        for nb in nb_intervals:
            df = self.exp.get_df(variable_name).loc[pd.IndexSlice[:, :, nb], :].dropna()
            values = df.values

            if len(df) > 100 and len(set(df.index.get_level_values(id_exp_name))) > 2:
                fisher_info = compute_fisher_information_uniform_von_mises(values)
                # fisher_info = 1 / np.var(values)

                self.exp.get_df(info_result_name).loc[nb, :] = np.around(fisher_info, 5)

    def __compute_fisher_info_evol(self, start_time_intervals, end_time_intervals,
                                   variable_name, info_result_name):
        nb_intervals = set(self.exp.get_index(variable_name).get_level_values('nb_attachments'))
        for nb in nb_intervals:
            for i in range(len(start_time_intervals)):
                frame0 = start_time_intervals[i]
                frame1 = end_time_intervals[i]
                column_name = str([frame0 / 100., frame1 / 100.])

                df = self.exp.get_df(variable_name).loc[pd.IndexSlice[:, frame0:frame1, nb], :].dropna()
                if len(df) > 100 and len(set(df.index.get_level_values(id_exp_name))) > 2:
                    values = df.values
                    fisher_info = compute_fisher_information_uniform_von_mises(values)
                    # fisher_info = 1 / np.var(values)

                    self.exp.get_df(info_result_name).loc[nb, column_name] = np.around(fisher_info, 5)

    def compute_foodvelocity_foodantvector_angle_around_outside_manual_leader_attachments(self):
        attachment_name = 'outside_manual_leading_attachments'
        variable_name = 'foodVelocity_foodAntVector_angle'

        result_name = variable_name + '_around_outside_manual_leading_attachments'

        self.exp.load(attachment_name)
        df = self.exp.get_df(attachment_name).copy()
        df = df[df == 1].dropna()
        self.exp.change_df(attachment_name, df)

        result_label = 'Angle between food velocity and food-ant vector' \
                       ' around manual leading outside attachments'
        result_description = 'Angle between food velocity and food-ant vector for times' \
                             ' before and after an ant coming from outside ant attached to the food,' \
                             ' the attachments have been manually identified has leading'

        self.__gather_exp_ant_frame_indexed_around_attachments(
            variable_name, attachment_name, result_name, result_label, result_description)
        self.exp.remove_object(attachment_name)

    def compute_information_foodvelocity_foodantvector_angle_around_outside_manual_leading_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'foodVelocity_foodAntVector_angle_around_outside_manual_leading_attachments'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around manual outside attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around outside ant attachments,' \
                            ' the attachments have been manually identified has leading'

        info_label = 'Information of the food around outside attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside ant attachments,' \
                           ' the attachments have been manually identified has leading'

        ylim_zoom = (0.2, 2)
        dpi = 1/12.
        self.__compute_information_around_attachments(self.__compute_entropy, dpi, variable_name, hists_result_name,
                                                      info_result_name, hists_label, hists_description,
                                                      info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

    def compute_foodvelocity_foodantvector_angle_around_inside_manual_leader_attachments(self):
        attachment_name = 'inside_manual_leading_attachments'
        variable_name = 'foodVelocity_foodAntVector_angle'

        result_name = variable_name + '_around_inside_manual_leading_attachments'

        self.exp.load(attachment_name)
        df = self.exp.get_df(attachment_name).copy()
        df = df[df == 1].dropna()
        self.exp.change_df(attachment_name, df)

        result_label = 'Angle between food velocity and food-ant vector' \
                       ' around manual leading outside attachments'
        result_description = 'Angle between food velocity and food-ant vector for times' \
                             ' before and after an ant coming from outside ant attached to the food,' \
                             ' the attachments have been manually identified has leading'

        self.__gather_exp_ant_frame_indexed_around_attachments(
            variable_name, attachment_name, result_name, result_label, result_description)
        self.exp.remove_object(attachment_name)

    def compute_information_foodvelocity_foodantvector_angle_around_inside_manual_leading_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'foodVelocity_foodAntVector_angle_around_inside_manual_leading_attachments'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around manual outside attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around outside ant attachments,' \
                            ' the attachments have been manually identified has leading'

        info_label = 'Information of the food around outside attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside ant attachments,' \
                           ' the attachments have been manually identified has leading'

        ylim_zoom = (0.2, 1)
        dpi = 1/12.
        self.__compute_information_around_attachments(self.__compute_entropy, dpi, variable_name, hists_result_name,
                                                      info_result_name, hists_label, hists_description,
                                                      info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

    def compute_foodvelocity_foodantvector_angle_around_outside_manual_follower_attachments(self):
        attachment_name = 'outside_manual_leading_attachments'
        variable_name = 'foodVelocity_foodAntVector_angle'

        result_name = variable_name + '_around_outside_manual_following_attachments'

        self.exp.load(attachment_name)
        df = self.exp.get_df(attachment_name).copy()
        df = df[df == 0].dropna()
        self.exp.change_df(attachment_name, df)

        result_label = 'Angle between food velocity and food-ant vector' \
                       ' around manual leading outside attachments'
        result_description = 'Angle between food velocity and food-ant vector for times' \
                             ' before and after an ant coming from outside ant attached to the food,' \
                             ' the attachments have been manually identified has leading'

        self.__gather_exp_ant_frame_indexed_around_attachments(
            variable_name, attachment_name, result_name, result_label, result_description)
        self.exp.remove_object(attachment_name)

    def compute_information_foodvelocity_foodantvector_angle_around_outside_manual_following_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'foodVelocity_foodAntVector_angle_around_outside_manual_following_attachments'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around manual outside attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around outside ant attachments,' \
                            ' the attachments have been manually identified has leading'

        info_label = 'Information of the food around outside attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside ant attachments,' \
                           ' the attachments have been manually identified has leading'

        ylim_zoom = (0.1, 0.3)
        dpi = 1/12.
        self.__compute_information_around_attachments(self.__compute_entropy, dpi, variable_name, hists_result_name,
                                                      info_result_name, hists_label, hists_description,
                                                      info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

    def compute_foodvelocity_foodantvector_angle_around_inside_manual_follower_attachments(self):
        attachment_name = 'inside_manual_leading_attachments'
        variable_name = 'foodVelocity_foodAntVector_angle'

        result_name = variable_name + '_around_inside_manual_following_attachments'

        self.exp.load(attachment_name)
        df = self.exp.get_df(attachment_name).copy()
        df = df[df == 0].dropna()
        self.exp.change_df(attachment_name, df)

        result_label = 'Angle between food velocity and food-ant vector' \
                       ' around manual leading outside attachments'
        result_description = 'Angle between food velocity and food-ant vector for times' \
                             ' before and after an ant coming from outside ant attached to the food,' \
                             ' the attachments have been manually identified has leading'

        self.__gather_exp_ant_frame_indexed_around_attachments(
            variable_name, attachment_name, result_name, result_label, result_description)
        self.exp.remove_object(attachment_name)

    def compute_information_foodvelocity_foodantvector_angle_around_inside_manual_following_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'foodVelocity_foodAntVector_angle_around_inside_manual_following_attachments'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around manual outside attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around outside ant attachments,' \
                            ' the attachments have been manually identified has leading'

        info_label = 'Information of the food around outside attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside ant attachments,' \
                           ' the attachments have been manually identified has leading'

        ylim_zoom = (0.1, 0.3)
        dpi = 1/12.
        self.__compute_information_around_attachments(self.__compute_entropy, dpi, variable_name, hists_result_name,
                                                      info_result_name, hists_label, hists_description,
                                                      info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

    def __gather_exp_ant_frame_indexed_around_attachments(
            self, variable_name, attachment_name, result_name, result_label, result_description):

        last_frame_name = 'food_exit_frames'
        self.exp.load([variable_name, last_frame_name, 'fps'])

        t0, t1, dt = -60, 60, 0.1
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)

        index_values = self.exp.get_df(attachment_name).reset_index()
        index_values = index_values.set_index([id_exp_name, id_frame_name])
        index_values = index_values.sort_index()
        index_values = index_values.index
        self.exp.add_new_empty_dataset(name=result_name, index_names=[id_exp_name, id_frame_name],
                                       column_names=np.array(time_intervals, dtype=str),
                                       index_values=index_values,
                                       category=self.category, label=result_label, description=result_description)

        def get_variable4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            fps = self.exp.get_value('fps', id_exp)
            last_frame = self.exp.get_value(last_frame_name, id_exp)

            if id_exp in self.exp.get_index(attachment_name).get_level_values(id_exp_name):
                idx = self.exp.get_df(attachment_name).loc[id_exp, :, :].index
                for id_exp, attach_frame, id_ant in idx:
                    if attach_frame < last_frame:
                        print(id_exp, attach_frame)
                        f0 = int(attach_frame + time_intervals[0] * fps)
                        f1 = int(attach_frame + time_intervals[-1] * fps)

                        var_df = df.loc[pd.IndexSlice[id_exp, id_ant, f0:f1], :]
                        var_df = var_df.loc[id_exp, id_ant, :]
                        var_df.index -= attach_frame
                        var_df.index /= fps

                        var_df = var_df.reindex(time_intervals)

                        self.exp.get_df(result_name).loc[(id_exp, attach_frame), :] = np.array(var_df[variable_name])

        self.exp.groupby(variable_name, id_exp_name, func=get_variable4each_group)
        self.exp.write(result_name)
