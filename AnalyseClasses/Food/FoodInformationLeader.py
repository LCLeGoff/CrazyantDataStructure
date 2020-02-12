import numpy as np
import pandas as pd
import random as rd

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_frame_name, id_ant_name
from Tools.MiscellaneousTools.ArrayManipulation import get_max_entropy, get_entropy2
from Tools.Plotter.ColorObject import ColorObject
from Tools.Plotter.Plotter import Plotter


class AnalyseFoodInformationLeader(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'FoodInformationLeader'

    def __compute_entropy(self, time_intervals, hists_result_name, info_result_name):
        for t in time_intervals:
            hist = self.exp.get_df(hists_result_name)[t]
            entropy, v = get_entropy2(hist, get_variance=True)
            max_entropy = get_max_entropy(hist)

            info = np.around(max_entropy - entropy, 6)
            self.exp.get_df(info_result_name).loc[t, 'info'] = info

            self.exp.get_df(info_result_name).loc[t, 'err1'] = 1.95 * np.sqrt(v)
            self.exp.get_df(info_result_name).loc[t, 'err2'] = 1.95 * np.sqrt(v)

    def __compute_entropy_evol(self, time_intervals, hists_result_name, info_result_name):
        column_names = self.exp.get_columns(info_result_name)
        for inter in column_names:
            for dt in time_intervals:
                hist = self.exp.get_df(hists_result_name).loc[pd.IndexSlice[dt, :], inter]
                entropy = get_entropy2(hist)
                max_entropy = get_max_entropy(hist)

                self.exp.get_df(info_result_name).loc[dt, inter] = np.around(max_entropy - entropy, 3)

    def __compute_fisher_info(self, time_intervals, variable_name, info_result_name):
        for t in time_intervals:
            print(t)
            values = np.array(self.exp.get_df(variable_name)[str(t)].dropna())
            fisher_info = round(1 / np.var(values), 5)

            lg = len(values)
            fisher_list = []
            for i in range(1000):
                idx = np.random.randint(0, lg, lg)
                val = values[idx]
                fisher_list.append(1 / np.var(val))

            self.exp.get_df(info_result_name).loc[t, 'info'] = fisher_info

            self.exp.get_df(info_result_name).loc[t, 'err1'] = round(fisher_info - np.percentile(fisher_list, 2.5), 5)
            self.exp.get_df(info_result_name).loc[t, 'err2'] = round(np.percentile(fisher_list, 97.5) - fisher_info, 5)

    def compute_mm1s_food_direction_error_around_outside_leader_attachments(self):
        attachment_name = 'outside_attachment_intervals'
        self.exp.load(attachment_name)
        variable_name = 'mm1s_food_direction_error'

        result_name = variable_name + '_around_outside_leader_attachments'

        result_label = 'Food direction error around outside leader attachments'
        result_description = 'Food direction error smoothed with a moving mean of window 1s for times' \
                             ' before and after an ant coming from outside ant attached to the food and has' \
                             ' an influence on it'

        self.__gather_exp_frame_indexed_around_leader_attachments(
            variable_name, attachment_name, result_name, result_label, result_description)

    def compute_mm1s_food_direction_error_around_inside_leader_attachments(self):
        attachment_name = 'inside_attachment_intervals'
        self.exp.load(attachment_name)
        variable_name = 'mm1s_food_direction_error'

        result_name = variable_name + '_around_inside_leader_attachments'

        result_label = 'Food direction error around inside leader attachments'
        result_description = 'Food direction error smoothed with a moving mean of window 1s for times' \
                             ' before and after an ant coming from inside ant attached to the food and has' \
                             ' an influence on it'

        self.__gather_exp_frame_indexed_around_leader_attachments(
            variable_name, attachment_name, result_name, result_label, result_description)

    def __gather_exp_frame_indexed_around_leader_attachments(
            self, variable_name, attachment_name, result_name, result_label, result_description):

        last_frame_name = 'food_exit_frames'
        first_frame_name = 'first_attachment_time_of_outside_ant'
        leader_name = 'is_leader'
        self.exp.load([variable_name, first_frame_name, last_frame_name, leader_name, 'fps'])

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

        leader_index = list(self.exp.get_index(leader_name).droplevel(id_ant_name))

        def get_variable4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            fps = self.exp.get_value('fps', id_exp)
            last_frame = self.exp.get_value(last_frame_name, id_exp)
            first_frame = self.exp.get_value(first_frame_name, id_exp)

            if id_exp in self.exp.get_index(attachment_name).get_level_values(id_exp_name):
                attachment_frames = self.exp.get_df(attachment_name).loc[id_exp, :, :]
                attachment_frames = list(set(attachment_frames.index.get_level_values(id_frame_name)))
                attachment_frames.sort()

                for attach_frame in attachment_frames:

                    if (id_exp, attach_frame) in leader_index:
                        is_leader = self.exp.get_df(leader_name).loc[id_exp, :, attach_frame]
                        is_leader = int(is_leader.iloc[0])

                        if is_leader == 1 and last_frame > attach_frame > first_frame:
                            f0 = int(attach_frame + time_intervals[0] * fps)
                            f1 = int(attach_frame + time_intervals[-1] * fps)

                            var_df = df.loc[pd.IndexSlice[id_exp, f0:f1], :]
                            var_df = var_df.loc[id_exp, :]
                            var_df.index -= attach_frame
                            var_df.index /= fps

                            var_df = var_df.reindex(time_intervals)

                            self.exp.get_df(result_name).loc[(id_exp, attach_frame), :] =\
                                np.array(var_df[variable_name])

        self.exp.groupby(variable_name, id_exp_name, func=get_variable4each_group)
        self.exp.change_df(result_name, self.exp.get_df(result_name).dropna(how='all'))
        self.exp.write(result_name)

    def compute_information_mm1s_food_direction_error_around_outside_leader_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'mm1s_food_direction_error_around_outside_leader_attachments'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around outside leader attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around outside leader ant attachments'

        info_label = 'Information of the food around outside leader attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside leader ant attachments'

        ylim_zoom = (0., 1)
        dpi = 1/12.
        self.__compute_information_around_leader_attachments(self.__compute_entropy, dpi, variable_name,
                                                             hists_result_name, info_result_name, hists_label,
                                                             hists_description, info_label, info_description, ylim_zoom,
                                                             redo, redo_info, redo_plot_hist)

    def compute_information_mm1s_food_direction_error_around_inside_leader_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'mm1s_food_direction_error_around_inside_leader_attachments'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around inside leader attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around inside leader ant attachments'

        info_label = 'Information of the food around inside leader attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around inside leader ant attachments'

        ylim_zoom = (0., 0.5)
        dpi = 1/12.
        self.__compute_information_around_leader_attachments(self.__compute_entropy, dpi, variable_name,
                                                             hists_result_name, info_result_name, hists_label,
                                                             hists_description, info_label, info_description, ylim_zoom,
                                                             redo, redo_info, redo_plot_hist)

    def compute_information_mm1s_food_direction_error_around_leader_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'mm1s_food_direction_error_around_outside_leader_attachments'
        variable_name2 = 'mm1s_food_direction_error_around_inside_leader_attachments'
        self.exp.load([variable_name, variable_name2])
        self.exp.get_data_object(variable_name).df =\
            pd.concat([self.exp.get_df(variable_name), self.exp.get_df(variable_name2)])

        hists_result_name = 'histograms_mm1s_food_direction_error_around_leader_attachments'
        info_result_name = 'information_mm1s_food_direction_error_around_leader_attachments'

        hists_label = 'Histograms of the food direction error around leader attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around leader ant attachments'

        info_label = 'Information of the food around outside leader attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around leader ant attachments'

        ylim_zoom = (0.2, .3)
        dpi = 1/12.
        self.__compute_information_around_leader_attachments(self.__compute_entropy, dpi, variable_name,
                                                             hists_result_name, info_result_name, hists_label,
                                                             hists_description, info_label, info_description, ylim_zoom,
                                                             redo, redo_info, redo_plot_hist)

    def compute_information_mm1s_food_direction_error_around_outside_leader_attachments_partial(
            self, redo=False, redo_info=False, list_exps=None):

        variable_name = 'mm1s_food_direction_error_around_outside_leader_attachments'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name+'_partial'
        info_result_name = 'information_' + variable_name+'_partial'

        hists_label = 'Histograms of the food direction error around outside leader attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around outside leader ant attachments for exp in list_exps'

        info_label = 'Information of the food around outside leader attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside leader ant attachments' \
                           ' for exp in list_exps'

        ylim_zoom = (0., 0.5)
        dpi = 1/12.
        self.__compute_information_around_leader_attachments(self.__compute_entropy, dpi, variable_name,
                                                             hists_result_name, info_result_name, hists_label,
                                                             hists_description, info_label, info_description, ylim_zoom,
                                                             redo, redo_info, False, list_exps=list_exps)

    def compute_information_mm1s_food_direction_error_around_inside_leader_attachments_partial(
            self, redo=False, redo_info=False, list_exps=None):

        variable_name = 'mm1s_food_direction_error_around_inside_leader_attachments'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name+'_partial'
        info_result_name = 'information_' + variable_name+'_partial'

        hists_label = 'Histograms of the food direction error around inside leader attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around inside leader ant attachments for exp in list_exps'

        info_label = 'Information of the food around inside leader attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around inside leader ant attachments' \
                           ' for exp in list_exps'

        ylim_zoom = (0., 0.5)
        dpi = 1/12.
        self.__compute_information_around_leader_attachments(self.__compute_entropy, dpi, variable_name,
                                                             hists_result_name, info_result_name, hists_label,
                                                             hists_description, info_label, info_description, ylim_zoom,
                                                             redo, redo_info, False, list_exps=list_exps)

    def __compute_information_around_leader_attachments(self, fct, dpi, variable_name, hists_result_name,
                                                        info_result_name, hists_label, hists_description, info_label,
                                                        info_description, ylim_zoom, redo, redo_info, redo_plot_hist,
                                                        list_exps=None):
        t0, t1, dt = -60, 60, .5
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)
        dtheta = np.pi * dpi
        bins = np.arange(0, np.pi + dtheta, dtheta)
        hists_index_values = np.around((bins[1:] + bins[:-1]) / 2., 3)
        if redo:

            self.exp.add_new_empty_dataset(name=hists_result_name, index_names='food_direction_error',
                                           column_names=time_intervals, index_values=hists_index_values,
                                           category=self.category, label=hists_label, description=hists_description,
                                           replace=True)

            print(hists_result_name, len(self.exp.get_df(variable_name)['0.0'].abs().dropna()))

            for t in time_intervals:
                values = self.exp.get_df(variable_name)[str(t)].abs().dropna()
                if list_exps is not None:
                    values = values.loc[list_exps, :]
                hist = np.histogram(values, bins=bins, normed=False)[0]

                self.exp.get_df(hists_result_name)[t] = hist

            self.exp.remove_object(variable_name)
            self.exp.write(hists_result_name)

        else:
            self.exp.load(hists_result_name)

        if redo or redo_info:
            time_intervals = self.exp.get_df(hists_result_name).columns

            self.exp.add_new_empty_dataset(name=info_result_name, index_names='time',
                                           column_names=['info', 'err1', 'err2'],
                                           index_values=time_intervals, category=self.category,
                                           label=info_label, description=info_description, replace=True)

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

        # if redo or redo_plot_hist:
        #     for t in time_intervals:
        #         plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(hists_result_name), column_name=t)
        #         fig, ax = plotter.plot(xlabel='Food direction error', ylabel='Probability')
        #         ax.set_ylim(ylim)
        #         plotter.save(fig, sub_folder=hists_result_name, name=t)

        self.exp.remove_object(info_result_name)
        self.exp.load(info_result_name)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.create_plot(figsize=(4.5, 8), nrows=2)
        plotter.plot_with_error(preplot=(fig, ax[0]), xlabel='time (s)', ylabel=ylabel, title='')
        plotter.plot_with_error(preplot=(fig, ax[1]), title='')

        ax[0].axvline(0, ls='--', c='k')
        ax[1].axvline(0, ls='--', c='k')
        ax[1].set_xlim((-2, 8))
        ax[1].set_ylim(ylim_zoom)
        plotter.save(fig)
        self.exp.remove_object(info_result_name)

    def __plot_info_evol(self, info_result_name, ylabel, ylim_zoom):

        self.exp.remove_object(info_result_name)
        self.exp.load(info_result_name)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel=ylabel, title='')

        ax.axvline(0, ls='--', c='k')
        ax.set_xlim((-2, 8))
        ax.set_ylim(ylim_zoom)
        plotter.save(fig)

    def __compute_information_around_leader_attachments_evol(
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

    def compute_information_mm1s_food_direction_error_around_outside_leader_attachments_evol(
            self, redo=False, redo_info=False):

        variable_name = 'mm1s_food_direction_error_around_outside_leader_attachments'
        self.exp.load(variable_name)
        init_frame_name = 'first_attachment_time_of_outside_ant'

        hists_result_name = 'histograms_' + variable_name+'_evol'
        info_result_name = 'information_' + variable_name+'_evol'

        hists_label = 'Histograms of the food direction error around outside leader attachments over time'
        hists_description = 'Time evolution of the histograms of the food direction error' \
                            ' for each time t in time_intervals, ' \
                            'which are times around outside leader ant attachments'

        info_label = 'Information of the food around outside leader attachments over time'
        info_description = 'Time evolution of the information of the food' \
                           ' (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside leader ant attachments'

        dx = 0.25
        dx2 = 0.5
        start_frame_intervals = np.arange(0, 1.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx2*60*100

        ylim_zoom = (0., 2)
        dpi = 1/12.
        self.__compute_information_around_leader_attachments_evol(
            self.__compute_entropy_evol, dpi,  start_frame_intervals, end_frame_intervals,
            variable_name, hists_result_name, info_result_name, init_frame_name,
            hists_label, hists_description, info_label, info_description, ylim_zoom, redo, redo_info)

    def compute_information_mm1s_food_direction_error_around_inside_leader_attachments_evol(
            self, redo=False, redo_info=False):

        variable_name = 'mm1s_food_direction_error_around_inside_leader_attachments'
        self.exp.load(variable_name)
        init_frame_name = 'first_attachment_time_of_outside_ant'

        hists_result_name = 'histograms_' + variable_name+'_evol'
        info_result_name = 'information_' + variable_name+'_evol'

        hists_label = 'Histograms of the food direction error around inside leader attachments over time'
        hists_description = 'Time evolution of the histograms of the food direction error' \
                            ' for each time t in time_intervals, ' \
                            'which are times around inside leader ant attachments'

        info_label = 'Information of the food around inside leader attachments over time'
        info_description = 'Time evolution of the information of the food' \
                           ' (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around inside leader ant attachments'

        dx = 0.25
        dx2 = 0.5
        start_frame_intervals = np.arange(0, 1.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx2*60*100

        ylim_zoom = (0., 2)
        dpi = 1/12.
        self.__compute_information_around_leader_attachments_evol(
            self.__compute_entropy_evol, dpi,  start_frame_intervals, end_frame_intervals,
            variable_name, hists_result_name, info_result_name, init_frame_name,
            hists_label, hists_description, info_label, info_description, ylim_zoom, redo, redo_info)

    def compute_information_mm1s_food_direction_error_around_leader_attachments_evol(
            self, redo=False, redo_info=False):

        variable_name = 'mm1s_food_direction_error_around_outside_leader_attachments'
        variable_name2 = 'mm1s_food_direction_error_around_inside_leader_attachments'
        self.exp.load([variable_name, variable_name2])
        self.exp.get_data_object(variable_name).df =\
            pd.concat([self.exp.get_df(variable_name), self.exp.get_df(variable_name2)])

        init_frame_name = 'first_attachment_time_of_outside_ant'

        hists_result_name = 'histograms_mm1s_food_direction_error_around_leader_attachments_evol'
        info_result_name = 'information_mm1s_food_direction_error_around_leader_attachments_evol'

        hists_label = 'Histograms of the food direction error around outside leader attachments over time'
        hists_description = 'Time evolution of the histograms of the food direction error' \
                            ' for each time t in time_intervals, ' \
                            'which are times around leader ant attachments'

        info_label = 'Information of the food around outside leader attachments over time'
        info_description = 'Time evolution of the information of the food' \
                           ' (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around leader ant attachments'

        dx = 0.25
        dx2 = 0.5
        start_frame_intervals = np.arange(0, 1.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx2*60*100

        ylim_zoom = (0., 2)
        dpi = 1/12.
        self.__compute_information_around_leader_attachments_evol(
            self.__compute_entropy_evol, dpi,  start_frame_intervals, end_frame_intervals,
            variable_name, hists_result_name, info_result_name, init_frame_name,
            hists_label, hists_description, info_label, info_description, ylim_zoom, redo, redo_info)

    def plot_inside_and_outside_leader_information(self):
        outside_name = 'information_mm1s_food_direction_error_around_outside_leader_attachments'
        inside_name = 'information_mm1s_food_direction_error_around_inside_leader_attachments'
        self.exp.load([outside_name, inside_name])
        result_name = 'information_outside_inside_leader_attachment'

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(outside_name))
        fig, ax = plotter.create_plot()
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='Time (s)', ylabel='Information (bit)', title='', label='outside', c='red')

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(inside_name))
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='Time (s)', ylabel='Information (bit)',
            title='', label='inside', c='navy')

        ax.set_xlim(-2, 8)
        ax.set_ylim(0, .8)
        plotter.draw_vertical_line(ax, label='attachment')
        plotter.draw_legend(ax)

        plotter.save(fig, name=result_name)

    def plot_inside_and_outside_leader_information_partial(self):
        outside_name = 'information_mm1s_food_direction_error_around_outside_leader_attachments_partial'
        inside_name = 'information_mm1s_food_direction_error_around_inside_leader_attachments_partial'
        self.exp.load([outside_name, inside_name])
        result_name = 'information_outside_inside_leader_attachment_partial'

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(outside_name))
        fig, ax = plotter.create_plot()
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='Time (s)', ylabel='Information (bit)', title='', label='outside', c='red')

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(inside_name))
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='Time (s)', ylabel='Information (bit)',
            title='', label='inside')

        ax.set_xlim(-2, 8)
        plotter.draw_vertical_line(ax, label='attachment')
        plotter.draw_legend(ax)

        plotter.save(fig, name=result_name)

    def compute_mm1s_food_direction_error_around_outside_leader_ordered_attachments(self):
        attachment_name = 'outside_attachment_intervals'
        self.exp.load(attachment_name)
        variable_name = 'mm1s_food_direction_error'

        result_name = variable_name + '_around_outside_leader_ordered_attachments'

        result_label = 'Food direction error around outside leader attachments'
        result_description = 'Food direction error smoothed with a moving mean of window 1s for times' \
                             ' before and after an ant coming from inside ant attached to the food and has' \
                             ' an influence on it'

        self.__gather_exp_ant_frame_indexed_around_leader_attachments(
            variable_name, attachment_name, result_name, result_label, result_description)

    def compute_information_mm1s_food_direction_error_around_outside_leader_ordered_attachments(
            self, redo=False, redo_info=False):

        variable_name = 'mm1s_food_direction_error_around_outside_leader_ordered_attachments'
        self.exp.load(variable_name)

        hists_result_name = 'histograms_' + variable_name
        info_result_name = 'information_' + variable_name

        hists_label = 'Histograms of the food direction error around outside leader attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around outside leader ant attachments'

        info_label = 'Information of the food around outside leader attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside leader ant attachments'

        ylim_zoom = (0., 1.)
        dpi = 1/12.
        self.__compute_information_around_leader_ordered_attachments(self.__compute_entropy_ordered, dpi, variable_name,
                                                                     hists_result_name, info_result_name, hists_label,
                                                                     hists_description, info_label, info_description,
                                                                     ylim_zoom, redo, redo_info)

    def __compute_information_around_leader_ordered_attachments(self, fct, dpi, variable_name, hists_result_name,
                                                                info_result_name, hists_label, hists_description,
                                                                info_label, info_description, ylim_zoom, redo,
                                                                redo_info):
        i_max = 2

        t0, t1, dt = -60, 60, 1.
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)
        dtheta = np.pi * dpi
        bins = np.arange(0, np.pi + dtheta, dtheta)

        hists_index_values = list(np.around((bins[1:] + bins[:-1]) / 2., 3))
        lg = len(hists_index_values)
        order_idx = [i for i in range(1, i_max+1) for _ in range(lg)]
        hists_index_values = list(zip(order_idx, hists_index_values*i_max))
        if redo:

            first_frame_name = 'first_attachment_time_of_outside_ant'
            self.exp.load(first_frame_name)

            self.exp.add_new_empty_dataset(name=hists_result_name, index_names=['order', 'food_direction_error'],
                                           column_names=time_intervals, index_values=hists_index_values,
                                           category=self.category, label=hists_label, description=hists_description,
                                           replace=True)

            self._add_order_index(variable_name)

            for i in range(1, i_max+1):
                print(i, len(self.exp.get_df(variable_name).loc[i, :]))

            for t in time_intervals:
                for i in range(1, i_max+1):
                    values = self.exp.get_df(variable_name).loc[i, str(t)].abs().dropna()
                    hist = np.histogram(values, bins=bins, normed=False)[0]
                    self.exp.get_df(hists_result_name).loc[(i, t)] = hist

                # values = list(self.exp.get_df(variable_name).loc[1, str(t)].abs().dropna())
                # lg = len(self.exp.get_df(variable_name).loc[2, :])
                # hist = np.histogram(rd.sample(values, k=lg), bins=bins, normed=False)[0]
                # self.exp.get_df(hists_result_name).loc[(1, t)] = hist
                #
                # values = self.exp.get_df(variable_name).loc[2, str(t)].abs().dropna()
                # hist = np.histogram(values, bins=bins, normed=False)[0]
                # self.exp.get_df(hists_result_name).loc[(2, t)] = hist

            self.exp.remove_object(variable_name)
            self.exp.write(hists_result_name)

        else:
            self.exp.load(hists_result_name)

        if redo or redo_info:
            time_intervals = list(self.exp.get_df(hists_result_name).columns)
            lg = len(time_intervals)
            order_idx = [i for i in range(1, i_max+1) for _ in range(lg)]
            time_intervals = list(zip(order_idx, time_intervals*i_max))

            self.exp.add_new_empty_dataset(name=info_result_name, index_names=['order', 'time'],
                                           column_names=['info', 'err1', 'err2'],
                                           index_values=time_intervals, category=self.category,
                                           label=info_label, description=info_description, replace=True)

            time_intervals = self.exp.get_df(hists_result_name).columns
            fct(time_intervals, hists_result_name, info_result_name)

            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)

        ylabel = 'Information (bit)'
        self.__plot_info_ordered(info_result_name, ylabel, ylim_zoom)

    def __gather_exp_ant_frame_indexed_around_leader_attachments(
            self, variable_name, attachment_name, result_name, result_label, result_description):

        last_frame_name = 'food_exit_frames'
        first_frame_name = 'first_attachment_time_of_outside_ant'
        leader_name = 'is_leader'
        self.exp.load([variable_name, first_frame_name, last_frame_name, leader_name, 'fps'])

        t0, t1, dt = -60, 60, 0.1
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)

        index_values = self.exp.get_df(attachment_name).index
        self.exp.add_new_empty_dataset(name=result_name, index_names=[id_exp_name, id_ant_name, id_frame_name],
                                       column_names=np.array(time_intervals, dtype=str),
                                       index_values=index_values,
                                       category=self.category, label=result_label, description=result_description)

        leader_index = list(self.exp.get_index(leader_name).droplevel(id_ant_name))

        def get_variable4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            fps = self.exp.get_value('fps', id_exp)
            last_frame = self.exp.get_value(last_frame_name, id_exp)
            first_frame = self.exp.get_value(first_frame_name, id_exp)

            if id_exp in self.exp.get_index(attachment_name).get_level_values(id_exp_name):
                attachment_frames = self.exp.get_df(attachment_name).loc[id_exp, :, :]
                attachment_frames = list(set(attachment_frames.index.get_level_values(id_frame_name)))
                attachment_frames.sort()

                for attach_frame in attachment_frames:
                    if (id_exp, attach_frame) in leader_index:
                        is_leader = self.exp.get_df(leader_name).loc[id_exp, :, attach_frame]
                        _, id_ant, _, is_leader = is_leader.reset_index().values[0]
                        is_leader = int(is_leader)
                        if is_leader == 1 and last_frame > attach_frame > first_frame:
                            f0 = int(attach_frame + time_intervals[0] * fps)
                            f1 = int(attach_frame + time_intervals[-1] * fps)

                            var_df = df.loc[pd.IndexSlice[id_exp, f0:f1], :]
                            var_df = var_df.loc[id_exp, :]
                            var_df.index -= attach_frame
                            var_df.index /= fps

                            var_df = var_df.reindex(time_intervals)

                            self.exp.get_df(result_name).loc[(id_exp, id_ant, attach_frame), :] =\
                                np.array(var_df[variable_name])

        self.exp.groupby(variable_name, id_exp_name, func=get_variable4each_group)
        self.exp.change_df(result_name, self.exp.get_df(result_name).dropna(how='all'))
        self.exp.write(result_name)

    def __compute_entropy_ordered(self, time_intervals, hists_result_name, info_result_name):
        i_max = int(max(self.exp.get_index(info_result_name).get_level_values('order')))

        for t in time_intervals:
            for i in range(1, i_max+1):
                hist = self.exp.get_df(hists_result_name).loc[i, :][t]
                entropy, v = get_entropy2(hist, get_variance=True)
                max_entropy = get_max_entropy(hist)
                info = np.around(max_entropy - entropy, 6)
                self.exp.get_df(info_result_name).loc[(i, t), 'info'] = info
                self.exp.get_df(info_result_name).loc[(i, t), 'err1'] = 1.95 * np.sqrt(v)
                self.exp.get_df(info_result_name).loc[(i, t), 'err2'] = 1.95 * np.sqrt(v)

    def __plot_info_ordered(self, info_result_name, ylabel, ylim_zoom):

        i_max = int(max(self.exp.get_index(info_result_name).get_level_values('order')))
        c = ['k', 'r', 'g', 'b']

        for i in range(1, i_max+1):
            self.exp.add_new_dataset_from_df(self.exp.get_df(info_result_name).loc[i, :], 'temp%i' % i,
                                             category=self.category, replace=True)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object('temp1'))
        fig, ax = plotter.create_plot()
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='time (s)', ylabel=ylabel, title='', label='First attach.', draw_lims=True)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object('temp2'))
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='time (s)',
            ylabel=ylabel, title='', label='Other attach.', c=c[2], draw_lims=True)

        # for i in range(2, i_max+1):
        #     plotter = Plotter(self.exp.root, obj=self.exp.get_data_object('temp%i' % i))
        #     plotter.plot_with_error(
        #         preplot=(fig, ax), xlabel='time (s)',
        #         ylabel=ylabel, title='', c=c[i], label='%ith' % i, draw_lims=True)

        plotter.draw_legend(ax)

        ax.axvline(0, ls='--', c='k')
        ax.set_xlim((-2, 8))
        ax.set_ylim(ylim_zoom)
        plotter.save(fig, name=info_result_name)
        self.exp.remove_object(info_result_name)

    def compute_information_mm1s_food_direction_error_around_outside_leader_ordered_random_attachments(
            self, redo=False):

        outside_variable_name = 'mm1s_food_direction_error_around_outside_leader_ordered_attachments'
        inside_variable_name = 'mm1s_food_direction_error_around_inside_leader_attachments'
        self.exp.load([outside_variable_name, inside_variable_name])

        hists_result_name = 'histograms_mm1s_food_direction_error_around_outside_leader_ordered_random_attachments'
        info_result_name = 'information_mm1s_food_direction_error_around_outside_leader_ordered_random_attachments'

        info_label = 'Information of the food around outside leader attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside leader ant attachments'

        ylim_zoom = (0., 1.)
        dpi = 1/6.
        self.__compute_information_around_leader_ordered_random_attachments(
            dpi, outside_variable_name, inside_variable_name, hists_result_name, info_result_name,
            info_label, info_description, ylim_zoom, redo)

    def __compute_information_around_leader_ordered_random_attachments(
            self, dpi, outside_variable_name, inside_variable_name, hists_result_name, info_result_name,
            info_label, info_description, ylim_zoom, redo):

        t0, t1, dt = -60, 60, 1.
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)
        dtheta = np.pi * dpi
        bins = np.arange(0, np.pi + dtheta, dtheta)

        hists_index_values = list(np.around((bins[1:] + bins[:-1]) / 2., 3))
        n = 500

        if redo:

            first_frame_name = 'first_attachment_time_of_outside_ant'
            self.exp.load(first_frame_name)

            self._add_order_index(outside_variable_name)

            self.exp.add_new_empty_dataset(name=hists_result_name, index_names='food_direction_error',
                                           column_names=time_intervals, index_values=hists_index_values, replace=True)
            for t in time_intervals:
                values = self.exp.get_df(outside_variable_name).loc[2, str(t)].abs().dropna()
                hist = np.histogram(values, bins=bins, normed=False)[0]
                self.exp.get_df(hists_result_name)[t] = hist

            time_intervals = list(np.array(self.exp.get_df(hists_result_name).columns, dtype=float))
            lg = len(time_intervals)
            order_idx = [i for i in [1, 2, 'in'] for _ in range(lg)]
            index = list(zip(order_idx, time_intervals*3))

            self.exp.add_new_empty_dataset(name=info_result_name, index_names=['order', 'time'],
                                           column_names=['info', 'err1', 'err2'],
                                           index_values=index, category=self.category,
                                           label=info_label, description=info_description, replace=True)

            for t in time_intervals:
                hist = self.exp.get_df(hists_result_name).loc[:, t]
                entropy, v = get_entropy2(hist, get_variance=True)
                max_entropy = get_max_entropy(hist)
                info = np.around(max_entropy - entropy, 6)
                self.exp.get_df(info_result_name).loc[(2, t), 'info'] = info
                self.exp.get_df(info_result_name).loc[(2, t), 'err1'] = 1.95 * np.sqrt(v)
                self.exp.get_df(info_result_name).loc[(2, t), 'err2'] = 1.95 * np.sqrt(v)

            lg1 = len(self.exp.get_df(outside_variable_name).loc[1, '0.0'])
            lg2 = len(self.exp.get_df(outside_variable_name).loc[2, '0.0'])

            hist_list = np.zeros((n, len(time_intervals)))
            for i in range(n):
                print(i)
                sample = rd.sample(range(lg1), k=lg2)
                for j, t in enumerate(time_intervals):
                    values = self.exp.get_df(outside_variable_name).loc[1, str(t)].abs()
                    values = list(values.iloc[sample].dropna())
                    hist = np.histogram(values, bins=bins, normed=False)[0]

                    entropy, v = get_entropy2(hist, get_variance=True)
                    max_entropy = get_max_entropy(hist)
                    info = np.around(max_entropy - entropy, 6)

                    hist_list[i, j] = info

            for i, t in enumerate(time_intervals):
                vals = hist_list[:, i]
                m = np.mean(vals)
                self.exp.get_df(info_result_name).loc[(1, t), 'info'] = m
                self.exp.get_df(info_result_name).loc[(1, t), 'err1'] = m-np.percentile(vals, 2.5)
                self.exp.get_df(info_result_name).loc[(1, t), 'err2'] = np.percentile(vals, 97.5)-m

            lg1 = len(self.exp.get_df(inside_variable_name)['0.0'])

            hist_list = np.zeros((n, len(time_intervals)))
            for i in range(n):
                print(i)
                sample = rd.sample(range(lg1), k=lg2)
                for j, t in enumerate(time_intervals):
                    values = self.exp.get_df(inside_variable_name)[str(t)].abs()
                    values = list(values.iloc[sample].dropna())
                    hist = np.histogram(values, bins=bins, normed=False)[0]

                    entropy, v = get_entropy2(hist, get_variance=True)
                    max_entropy = get_max_entropy(hist)
                    info = np.around(max_entropy - entropy, 6)

                    hist_list[i, j] = info

            for i, t in enumerate(time_intervals):
                vals = hist_list[:, i]
                m = np.mean(vals)
                self.exp.get_df(info_result_name).loc[('in', t), 'info'] = m
                self.exp.get_df(info_result_name).loc[('in', t), 'err1'] = m-np.percentile(vals, 2.5)
                self.exp.get_df(info_result_name).loc[('in', t), 'err2'] = np.percentile(vals, 97.5)-m

            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)
        self.exp.get_df(info_result_name).dropna(inplace=True)
        ylabel = 'Information (bit)'

        for i in [1, 2, 'in']:
            self.exp.add_new_dataset_from_df(self.exp.get_df(info_result_name).loc[i, :], 'temp%s' % str(i),
                                             category=self.category, replace=True)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object('temp1'))
        fig, ax = plotter.create_plot()
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='time (s)', ylabel=ylabel,
            title='', label='first outside attach.', draw_lims=True)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object('temp2'))
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='time (s)',
            ylabel=ylabel, title='', label='non-first outside attach.', c='g', draw_lims=True)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object('tempin'))
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='time (s)',
            ylabel=ylabel, title='', label='inside attach.', c='b', draw_lims=True)

        plotter.draw_legend(ax)

        ax.axvline(0, ls='--', c='k')
        ax.set_xlim((-2, 8))
        ax.set_ylim(ylim_zoom)
        plotter.save(fig, name=info_result_name)
        self.exp.remove_object(info_result_name)

    def _add_order_index(self, variable_name):

        self.exp.get_df(variable_name)['order'] = 1

        def do4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            id_ant = df.index.get_level_values(id_ant_name)[0]
            frames = list(df.index.get_level_values(id_frame_name))

            self.exp.get_df(variable_name).loc[pd.IndexSlice[id_exp, id_ant, frames[1:]], 'order'] = 2
            # if len(frames) == 1:
            #     self.exp.get_df(variable_name).drop(index=(id_exp, id_ant, frames[0]), inplace=True)

            # if len(frames) == i_max:
            #     for i in range(1, min(i_max, len(frames))):
            #         self.exp.get_df(variable_name).loc[pd.IndexSlice[id_exp, id_ant, frames[i]], 'order'] = i+1
            # else:
            #     self.exp.get_df(variable_name).drop(index=(id_exp, id_ant, frames[0]), inplace=True)

            return df

        self.exp.groupby(variable_name, [id_exp_name, id_ant_name], do4each_group)
        self.exp.get_df(variable_name).reset_index(inplace=True)
        self.exp.get_df(variable_name).set_index('order', inplace=True)
        self.exp.get_df(variable_name).drop(columns=id_exp_name, inplace=True)
        self.exp.get_df(variable_name).drop(columns=id_ant_name, inplace=True)
        self.exp.get_df(variable_name).drop(columns=id_frame_name, inplace=True)

    def compute_fisher_information_mm1s_food_direction_error_around_outside_leader_attachments(
            self, redo=False):

        variable_name = 'mm1s_food_direction_error_around_outside_leader_attachments'
        self.exp.load(variable_name)

        info_result_name = 'fisher_information_' + variable_name

        info_label = 'Fisher information of the food around outside leader attachments'
        info_description = 'Fisher information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside leader ant attachments'

        ylim_zoom = (0., 1)
        self.__compute_fisher_information_around_leader_attachments(
            self.__compute_fisher_info, variable_name, info_result_name, info_label, info_description,
            ylim_zoom, redo)

    def compute_fisher_information_mm1s_food_direction_error_around_inside_leader_attachments(
            self, redo=False):

        variable_name = 'mm1s_food_direction_error_around_inside_leader_attachments'
        self.exp.load(variable_name)

        info_result_name = 'fisher_information_' + variable_name

        info_label = 'Fisher information of the food around inside leader attachments'
        info_description = 'Fisher information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around inside leader ant attachments'

        ylim_zoom = (0., 1)
        self.__compute_fisher_information_around_leader_attachments(
            self.__compute_fisher_info, variable_name, info_result_name, info_label, info_description,
            ylim_zoom, redo)

    def __compute_fisher_information_around_leader_attachments(
            self, fct, variable_name, info_result_name, info_label, info_description,
            ylim_zoom, redo):

        t0, t1, dt = -10, 10, .5
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)
        if redo:

            self.exp.add_new_empty_dataset(name=info_result_name, index_names='time',
                                           column_names=['info', 'err1', 'err2'],
                                           index_values=time_intervals, category=self.category,
                                           label=info_label, description=info_description, replace=True)

            fct(time_intervals, variable_name, info_result_name)

            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)

        ylabel = 'Fisher information'
        ylim = (0, 0.25)

        self.__plot_info('', info_result_name, time_intervals, ylabel, ylim, ylim_zoom,
                         redo, False)

    def plot_inside_and_outside_leader_fisher_information(self):
        outside_name = 'fisher_information_mm1s_food_direction_error_around_outside_leader_attachments'
        inside_name = 'fisher_information_mm1s_food_direction_error_around_inside_leader_attachments'
        self.exp.load([outside_name, inside_name])
        result_name = 'fisher_information_outside_inside_leader_attachment'

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(outside_name))
        fig, ax = plotter.create_plot()
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='Time (s)', ylabel='fisher information', title='', label='outside', c='red')

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(inside_name))
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='Time (s)', ylabel='fisher information',
            title='', label='inside', c='navy')

        ax.set_xlim(-2, 8)
        ax.set_ylim(0.3, .8)
        plotter.draw_vertical_line(ax, label='attachment')
        plotter.draw_legend(ax)

        plotter.save(fig, name=result_name)

    def compute_fisher_information_mm1s_food_direction_error_around_outside_leader_attachments_evol(
            self, redo=False):

        variable_name = 'mm1s_food_direction_error_around_outside_leader_attachments'
        self.exp.load(variable_name)

        info_result_name = 'fisher_information_' + variable_name+'_evol'

        info_label = 'Fisher information of the food around outside leader attachments'
        info_description = 'Fisher information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside leader ant attachments'

        first_frame_name = 'first_attachment_time_of_outside_ant'
        self.change_first_frame(variable_name, first_frame_name)

        start_frame_intervals = np.array([0, ])*100
        end_frame_intervals = np.array([45, 140])*100

        ylim = (0.25, 1.25)

        self.__compute_fisher_information_around_leader_attachments_evol(
            self.__compute_fisher_info_evol, ylim, variable_name, info_result_name,
            start_frame_intervals, end_frame_intervals,
            info_label, info_description, redo)

    def compute_fisher_information_mm1s_food_direction_error_around_inside_leader_attachments_evol(
            self, redo=False):

        variable_name = 'mm1s_food_direction_error_around_inside_leader_attachments'
        self.exp.load(variable_name)

        info_result_name = 'fisher_information_' + variable_name+'_evol'

        info_label = 'Fisher information of the food around inside leader attachments'
        info_description = 'Fisher information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around inside leader ant attachments'

        first_frame_name = 'first_attachment_time_of_outside_ant'
        self.change_first_frame(variable_name, first_frame_name)

        dx = 0.5
        dx2 = 0.5
        start_frame_intervals = np.arange(0, 1.5, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx2*60*100

        ylim = (0.25, 1.25)

        self.__compute_fisher_information_around_leader_attachments_evol(
            self.__compute_fisher_info_evol, ylim, variable_name, info_result_name,
            start_frame_intervals, end_frame_intervals,
            info_label, info_description, redo)

    def __compute_fisher_information_around_leader_attachments_evol(
            self, fct, ylim, variable_name, info_result_name, start_frame_intervals, end_frame_intervals,
            info_label, info_description, redo):

        t0, t1, dt = -10, 10, .5
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)
        indexes = [(int(t1), t2) for t1 in start_frame_intervals for t2 in time_intervals]
        if redo:

            self.exp.add_new_empty_dataset(name=info_result_name, index_names=['time', 'time_evol'],
                                           column_names=['info', 'err1', 'err2'],
                                           index_values=indexes, category=self.category,
                                           label=info_label, description=info_description, replace=True)

            fct(time_intervals, variable_name, info_result_name, start_frame_intervals, end_frame_intervals)

            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)

        ylabel = 'Fisher information'

        self.__plot_info_evol(info_result_name, ylabel, ylim, start_frame_intervals, end_frame_intervals)

    def __compute_fisher_info_evol(
            self, time_intervals, variable_name, info_result_name, start_intervals, end_intervals):

        for t in time_intervals:
            print(t)
            for t1, t2 in list(zip(start_intervals, end_intervals)):
                values = np.array(self.exp.get_df(variable_name)[str(t)].loc[:, t1:t2].dropna())
                fisher_info = round(1 / np.var(values), 5)

                lg = len(values)
                fisher_list = []
                for i in range(1000):
                    idx = np.random.randint(0, lg, lg)
                    val = values[idx]
                    fisher_list.append(1 / np.var(val))

                q1 = np.percentile(fisher_list, 2.5)
                q2 = np.percentile(fisher_list, 97.5)

                self.exp.get_df(info_result_name).loc[(t1, t), 'info'] = fisher_info
                self.exp.get_df(info_result_name).loc[(t1, t), 'err1'] = round(fisher_info - q1, 5)
                self.exp.get_df(info_result_name).loc[(t1, t), 'err2'] = round(q2 - fisher_info, 5)

    def __plot_info_evol(self, info_result_name, ylabel, ylim, start_intervals, end_intervals):

        self.exp.remove_object(info_result_name)
        self.exp.load(info_result_name)

        cols = ColorObject.create_cmap('hot_r', start_intervals)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.create_plot()
        for i in range(len(start_intervals)):
            t1 = start_intervals[i]
            t2 = end_intervals[i]
            df = self.exp.get_df(info_result_name).loc[t1, :]
            self.exp.add_new_dataset_from_df(df, 'temp', replace=True)
            plotter2 = Plotter(self.exp.root, obj=self.exp.get_data_object('temp'))
            plotter2.plot_with_error(
                xlabel='Time (s)', ylabel=ylabel, title='', preplot=(fig, ax),
                label=r'$t\in[%.1f, %.1f[$s' % (t1/100., t2/100.), c=cols[str(t1)], draw_lims=True
            )

        ax.axvline(0, ls='--', c='k')
        ax.set_xlim((-2, 8))
        ax.set_ylim(ylim)
        plotter.draw_legend(ax)
        plotter.save(fig)

    def compute_fisher_information_mm1s_food_direction_error_around_outside_leader_attachments_split(
            self, redo=False):

        variable_name = 'mm1s_food_direction_error_around_outside_leader_attachments'
        self.exp.load(variable_name)

        info_result_name = 'fisher_information_' + variable_name+'_split'

        info_label = 'Fisher information of the food around outside leader attachments'
        info_description = 'Fisher information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around outside leader ant attachments'

        index = []

        def do4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            frames = np.array(df.index.get_level_values(id_frame_name)).ravel()
            frames[1:] -= frames[:-1]
            frames[0] = 20000
            index.append(list(zip([id_exp]*len(frames), frames)))

            return df

        self.exp.groupby(variable_name, id_exp_name, do4each_group)

        index2 = []
        for l in index:
            index2 += l

        index2 = pd.MultiIndex.from_tuples(index2, names=[id_exp_name, id_frame_name])

        self.exp.get_df(variable_name).index = index2

        ylim = (0.25, 1.25)
        labels = ['short', 'long']

        self.__compute_fisher_information_around_leader_attachments_split(
            self.__compute_fisher_info_split, labels, ylim, variable_name, info_result_name,
            info_label, info_description, redo)

    def __compute_fisher_information_around_leader_attachments_split(
            self, fct, labels, ylim, variable_name, info_result_name, info_label, info_description, redo):

        t0, t1, dt = -10, 10, .5
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)
        indexes = [(cat, t) for cat in labels for t in time_intervals]
        if redo:

            self.exp.add_new_empty_dataset(name=info_result_name, index_names=['time', 'cat'],
                                           column_names=['info', 'err1', 'err2'],
                                           index_values=indexes, category=self.category,
                                           label=info_label, description=info_description, replace=True)

            fct(time_intervals, variable_name, info_result_name, labels)

            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)

        ylabel = 'Fisher information'

        self.__plot_info_split(info_result_name, ylabel, ylim, labels)

    def __compute_fisher_info_split(
            self, time_intervals, variable_name, info_result_name, labels):

        self.exp.get_df(variable_name).sort_index(inplace=True)
        times = np.array(self.exp.get_index(variable_name).get_level_values(id_frame_name))
        med = int(np.median(times))
        # med = 5000
        maxi = int(np.max(times))
        for t in time_intervals:
            print(t)
            for i, (t1, t2) in enumerate([(0, med), (med, maxi+1)]):
                values = np.array(self.exp.get_df(variable_name)[str(t)].loc[:, t1:t2].dropna())
                fisher_info = round(1 / np.var(values), 5)

                lg = len(values)
                fisher_list = []
                for _ in range(1000):
                    idx = np.random.randint(0, lg, lg)
                    val = values[idx]
                    fisher_list.append(1 / np.var(val))

                q1 = np.percentile(fisher_list, 2.5)
                q2 = np.percentile(fisher_list, 97.5)

                self.exp.get_df(info_result_name).loc[(labels[i], t), 'info'] = fisher_info
                self.exp.get_df(info_result_name).loc[(labels[i], t), 'err1'] = round(fisher_info - q1, 5)
                self.exp.get_df(info_result_name).loc[(labels[i], t), 'err2'] = round(q2 - fisher_info, 5)

    def __plot_info_split(self, info_result_name, ylabel, ylim, labels):

        self.exp.remove_object(info_result_name)
        self.exp.load(info_result_name)

        cols = ColorObject.create_cmap('hot_r', labels)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.create_plot()
        for cat in labels:
            df = self.exp.get_df(info_result_name).loc[cat, :]
            self.exp.add_new_dataset_from_df(df, 'temp', replace=True)
            plotter2 = Plotter(self.exp.root, obj=self.exp.get_data_object('temp'))
            plotter2.plot_with_error(
                xlabel='Time (s)', ylabel=ylabel, title='', preplot=(fig, ax),
                label=cat, c=cols[cat], draw_lims=True
            )

        ax.axvline(0, ls='--', c='k')
        ax.set_xlim((-2, 8))
        ax.set_ylim(ylim)
        plotter.draw_legend(ax)
        plotter.save(fig)
