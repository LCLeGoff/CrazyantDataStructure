import numpy as np
import pandas as pd
import random as rd

from matplotlib.ticker import MultipleLocator

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_frame_name, id_ant_name
from Tools.MiscellaneousTools.ArrayManipulation import get_max_entropy, get_entropy2
from Tools.MiscellaneousTools.FoodFisherInformation import compute_fisher_information_uniform_von_mises, \
    compute_fisher_information_uniform_von_mises_fix
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

            self.exp.get_df(info_result_name).loc[t, 'CI95_1'] = 1.95 * np.sqrt(v)
            self.exp.get_df(info_result_name).loc[t, 'CI95_2'] = 1.95 * np.sqrt(v)

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
            values = np.array(self.exp.get_df(variable_name).loc[pd.IndexSlice[:], str(t-0.5):str(t+0.5)].dropna())
            # values = np.array(self.exp.get_df(variable_name)[str(t)].dropna())
            fisher_info, q, kappa, cov_mat = compute_fisher_information_uniform_von_mises(values, get_fit_quality=True)
            # fisher_info = round(1 / np.var(values), 5)
            q_err, kappa_err = 1.96*np.sqrt(np.diagonal(cov_mat))

            q1 = max(q - q_err, 0)
            q2 = min(q + q_err, 1)
            kappa1 = max(kappa - kappa_err, 0)
            kappa2 = kappa + kappa_err
            fi1 = compute_fisher_information_uniform_von_mises_fix(q1, kappa1)
            fi2 = compute_fisher_information_uniform_von_mises_fix(q2, kappa2)

            self.exp.get_df(info_result_name).loc[t, 'info'] = fisher_info
            self.exp.get_df(info_result_name).loc[t, 'CI95_2'] = fi2-fisher_info
            self.exp.get_df(info_result_name).loc[t, 'CI95_1'] = fisher_info-fi1

            # lg = len(values)
            # fisher_list = []
            # i = 0
            # while i < 1000:
            #     try:
            #         idx = np.random.randint(0, lg, lg)
            #         val = values[idx]
            #         fi = compute_fisher_information_uniform_von_mises(val)
            #         # fisher_list.append(1 / np.var(val))
            #         fisher_list.append(fi)
            #         i += 1
            #     except RuntimeError:
            #         pass

            # self.exp.get_df(info_result_name).loc[t, 'info'] = fisher_info
            # ci95 = round(fisher_info - np.percentile(fisher_list, 2.5), 5)
            # self.exp.get_df(info_result_name).loc[t, 'CI95_1'] = ci95
            # ci95 = round(np.percentile(fisher_list, 97.5) - fisher_info, 5)
            # self.exp.get_df(info_result_name).loc[t, 'CI95_2'] = ci95

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

    def compute_mm1s_food_direction_error_around_isolated_leading_attachments(self):
        attachment_name = 'isolated_%s_leading_attachment_intervals'
        variable_name = 'mm1s_food_direction_error'

        result_name = variable_name + '_around_isolated_%s_leader_attachments'

        result_label = 'Food direction error around isolated %s leader attachments'
        result_description = 'Food direction error smoothed with a moving mean of window 1s for times' \
                             ' before and after an ant coming from %s ant attached to the food and has' \
                             ' an influence on it and that no other leading attachment occurred 2s after and before'

        for suff in ['outside', 'inside']:
            self.exp.load(attachment_name % suff)
            self.__gather_exp_frame_indexed_around_leader_attachments(
                variable_name, attachment_name % suff,
                result_name % suff, result_label % suff, result_description % suff)

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
            self, variable_name, attachment_name, result_name, result_label, result_description,
            time_min=None, time_max=None):

        if time_min is None:
            time_min = -np.inf

        if time_max is None:
            time_max = np.inf

        last_frame_name = 'food_exit_frames'
        first_frame_name = 'first_attachment_time_of_outside_ant'
        leader_name = 'is_leader'
        self.exp.load([variable_name, first_frame_name, last_frame_name, leader_name, 'fps'], reload=False)

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

        leader_index = list(self.exp.get_index(leader_name).droplevel(id_ant_name))

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

                            self.exp.get_df(result_name).loc[(id_exp, attach_frame), :] = \
                                np.array(var_df[variable_name])

        self.exp.groupby(variable_name, id_exp_name, func=get_variable4each_group)
        self.exp.change_df(result_name, self.exp.get_df(result_name).dropna(how='all'))
        print(len(self.exp.get_df(result_name)))
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
        self.exp.get_data_object(variable_name).df = \
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
                                           column_names=['info', 'CI95_1', 'CI95_2'],
                                           index_values=time_intervals, category=self.category,
                                           label=info_label, description=info_description, replace=True)

            fct(time_intervals, hists_result_name, info_result_name)

            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)

        ylabel = 'Information (bit)'
        ylim = None

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
        self.exp.get_data_object(variable_name).df = \
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
        m = []
        m += list(self.exp.get_df(outside_name).loc[-2:0, 'info'])
        m += list(self.exp.get_df(inside_name).loc[-2:0, 'info'])
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
                                           column_names=['info', 'CI95_1', 'CI95_2'],
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
        self.exp.load([variable_name, attachment_name, first_frame_name, last_frame_name, leader_name, 'fps'])

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

            if id_exp in set(self.exp.get_index(attachment_name).get_level_values(id_exp_name)):
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

                            self.exp.get_df(result_name).loc[(id_exp, id_ant, attach_frame), :] = \
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
                self.exp.get_df(info_result_name).loc[(i, t), 'CI95_1'] = 1.95 * np.sqrt(v)
                self.exp.get_df(info_result_name).loc[(i, t), 'CI95_2'] = 1.95 * np.sqrt(v)

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
            self, dpi, typ, variable_name, hists_result_name, info_result_name,
            info_label, info_description, ylim_zoom, redo):

        self.exp.load(variable_name)

        t0, t1, dt = -60, 60, 1.
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)
        dtheta = np.pi * dpi
        bins = np.arange(0, np.pi + dtheta, dtheta)

        hists_index_values = list(np.around((bins[1:] + bins[:-1]) / 2., 3))
        n = 500

        if redo:

            first_frame_name = 'first_attachment_time_of_outside_ant'
            self.exp.load(first_frame_name)

            self._add_order_index(variable_name)

            self.exp.add_new_empty_dataset(name=hists_result_name, index_names='food_direction_error',
                                           column_names=time_intervals, index_values=hists_index_values, replace=True)
            for t in time_intervals:
                values = self.exp.get_df(variable_name).loc[2, str(t)].abs().dropna()
                hist = np.histogram(values, bins=bins, normed=False)[0]
                self.exp.get_df(hists_result_name)[t] = hist

            time_intervals = list(np.array(self.exp.get_df(hists_result_name).columns, dtype=float))
            lg = len(time_intervals)
            order_idx = [i for i in [1, 2] for _ in range(lg)]
            index = list(zip(order_idx, time_intervals*2))

            self.exp.add_new_empty_dataset(name=info_result_name, index_names=['order', 'time'],
                                           column_names=['info', 'CI95_1', 'CI95_2'],
                                           index_values=index, category=self.category,
                                           label=info_label, description=info_description, replace=True)

            for t in time_intervals:
                hist = self.exp.get_df(hists_result_name).loc[:, t]
                entropy, v = get_entropy2(hist, get_variance=True)
                max_entropy = get_max_entropy(hist)
                info = np.around(max_entropy - entropy, 6)
                self.exp.get_df(info_result_name).loc[(2, t), 'info'] = info
                self.exp.get_df(info_result_name).loc[(2, t), 'CI95_1'] = 1.95 * np.sqrt(v)
                self.exp.get_df(info_result_name).loc[(2, t), 'CI95_2'] = 1.95 * np.sqrt(v)

            lg1 = len(self.exp.get_df(variable_name).loc[1, '0.0'])
            lg2 = len(self.exp.get_df(variable_name).loc[2, '0.0'])

            hist_list = np.zeros((n, len(time_intervals)))
            for i in range(n):
                print(i)
                sample = rd.sample(range(lg1), k=lg2)
                for j, t in enumerate(time_intervals):
                    values = self.exp.get_df(variable_name).loc[1, str(t)].abs()
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
                self.exp.get_df(info_result_name).loc[(1, t), 'CI95_1'] = m-np.percentile(vals, 2.5)
                self.exp.get_df(info_result_name).loc[(1, t), 'CI95_2'] = np.percentile(vals, 97.5)-m

            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)
        self.exp.get_df(info_result_name).dropna(inplace=True)
        ylabel = 'Information (bit)'

        if typ == 'outside':
            c1 = 'red'
            c2 = 'orange'
        else:
            c1 = 'navy'
            c2 = 'blue'

        for i in [1, 2]:
            self.exp.add_new_dataset_from_df(self.exp.get_df(info_result_name).loc[i, :], 'temp%s' % str(i),
                                             category=self.category, replace=True)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object('temp1'))
        fig, ax = plotter.create_plot()
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='time (s)', ylabel=ylabel, c=c1,
            title='', label='first %s attach.' % typ, draw_lims=True)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object('temp2'))
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='time (s)',
            ylabel=ylabel, title='', label='non-first %s attach.' % typ, c=c2, draw_lims=True)

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
            ylim_zoom, redo, plot=True):

        t0, t1, dt = -9.5, 10, .5
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)
        if redo:

            self.exp.add_new_empty_dataset(name=info_result_name, index_names='time',
                                           column_names=['info', 'CI95_1', 'CI95_2'],
                                           index_values=time_intervals, category=self.category,
                                           label=info_label, description=info_description, replace=True)

            fct(time_intervals, variable_name, info_result_name)

            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)

        if plot is True:
            ylabel = 'Fisher information'
            ylim = (0, 0.25)

            self.__plot_info('', info_result_name, time_intervals, ylabel, ylim, ylim_zoom,
                             False, False)

    def plot_inside_and_outside_leader_fisher_information(self):
        outside_name = 'fisher_information_mm1s_food_direction_error_around_outside_leader_attachments'
        inside_name = 'fisher_information_mm1s_food_direction_error_around_inside_leader_attachments'
        self.exp.load([outside_name, inside_name])
        result_name = 'fisher_information_outside_inside_leader_attachment'

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(outside_name))
        fig, ax = plotter.create_plot()
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='Time (s)', ylabel=r'Fisher information (rad$^{-2}$)',
            title='', label='outside', c='red', bar_width=2.7)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(inside_name))
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='Time (s)', ylabel=r'Fisher information (rad$^{-2}$)',
            title='', label='inside', c='navy')

        ax.set_xlim(-2, 8)
        ax.set_ylim(0., 1.25)
        plotter.draw_vertical_line(ax, label='attachment')
        m = []
        m += list(self.exp.get_df(outside_name).loc[-2:0.5, 'info'])[:-1]
        m += list(self.exp.get_df(inside_name).loc[-2:0.5, 'info'])[:-1]
        y0 = np.around(np.mean(m), 2)
        plotter.draw_horizontal_line(ax, val=y0, c='w', label='y= %.2f' % y0)
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

        ylim = None

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

        ylim = None

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
                                           column_names=['info', 'CI95_1', 'CI95_2'],
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
                # fisher_info = round(1 / np.var(values), 5)
                # fisher_info = compute_fisher_information_uniform_von_mises(values)

                fisher_info, q, kappa, cov_mat = compute_fisher_information_uniform_von_mises(
                    values, get_fit_quality=True)
                q_err, kappa_err = 1.96 * np.sqrt(np.diagonal(cov_mat))

                q1 = max(q - q_err, 0)
                q2 = min(q + q_err, 1)
                kappa1 = max(kappa - kappa_err, 0)
                kappa2 = kappa + kappa_err
                fi1 = compute_fisher_information_uniform_von_mises_fix(q1, kappa1)
                fi2 = compute_fisher_information_uniform_von_mises_fix(q2, kappa2)

                self.exp.get_df(info_result_name).loc[t, 'info'] = fisher_info
                self.exp.get_df(info_result_name).loc[t, 'CI95_2'] = fi2 - fisher_info
                self.exp.get_df(info_result_name).loc[t, 'CI95_1'] = fisher_info - fi1

                # lg = len(values)
                # fisher_list = []
                #
                # i = 0
                # while i < 500:
                #     try:
                #         idx = np.random.randint(0, lg, lg)
                #         val = values[idx]
                #         fi = compute_fisher_information_uniform_von_mises(val)
                #         # fisher_list.append(1 / np.var(val))
                #         fisher_list.append(fi)
                #         i += 1
                #     except RuntimeError:
                #         pass
                #
                # q1 = np.percentile(fisher_list, 2.5)
                # q2 = np.percentile(fisher_list, 97.5)
                #
                # self.exp.get_df(info_result_name).loc[(t1, t), 'info'] = fisher_info
                # self.exp.get_df(info_result_name).loc[(t1, t), 'CI95_1'] = round(fisher_info - q1, 5)
                # self.exp.get_df(info_result_name).loc[(t1, t), 'CI95_2'] = round(q2 - fisher_info, 5)

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
        for l1 in index:
            index2 += l1

        index2 = pd.MultiIndex.from_tuples(index2, names=[id_exp_name, id_frame_name])

        self.exp.get_df(variable_name).index = index2

        ylim = None
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
                                           column_names=['info', 'CI95_1', 'CI95_2'],
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
                # fisher_info = compute_fisher_information_uniform_von_mises(values)
                # fisher_info = round(1 / np.var(values), 5)

                fisher_info, q, kappa, cov_mat = compute_fisher_information_uniform_von_mises(
                    values, get_fit_quality=True)
                q_err, kappa_err = 1.96 * np.sqrt(np.diagonal(cov_mat))

                q1 = max(q - q_err, 0)
                q2 = min(q + q_err, 1)
                kappa1 = max(kappa - kappa_err, 0)
                kappa2 = kappa + kappa_err
                fi1 = compute_fisher_information_uniform_von_mises_fix(q1, kappa1)
                fi2 = compute_fisher_information_uniform_von_mises_fix(q2, kappa2)

                self.exp.get_df(info_result_name).loc[t, 'info'] = fisher_info
                self.exp.get_df(info_result_name).loc[t, 'CI95_2'] = fi2 - fisher_info
                self.exp.get_df(info_result_name).loc[t, 'CI95_1'] = fisher_info - fi1
                # lg = len(values)
                # fisher_list = []
                # for _ in range(1000):
                #     idx = np.random.randint(0, lg, lg)
                #     val = values[idx]
                #     fi = compute_fisher_information_uniform_von_mises(val)
                #     # fisher_list.append(1 / np.var(val))
                #     fisher_list.append(fi)
                #
                # q1 = np.percentile(fisher_list, 2.5)
                # q2 = np.percentile(fisher_list, 97.5)
                #
                # self.exp.get_df(info_result_name).loc[(labels[i], t), 'info'] = fisher_info
                # self.exp.get_df(info_result_name).loc[(labels[i], t), 'CI95_1'] = round(fisher_info - q1, 5)
                # self.exp.get_df(info_result_name).loc[(labels[i], t), 'CI95_2'] = round(q2 - fisher_info, 5)

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

    def compute_fisher_information_mm1s_food_direction_error_around_isolated_leader_attachments(
            self, redo=False):

        variable_name = 'mm1s_food_direction_error_around_isolated_%s_leader_attachments'
        info_label = 'Fisher information of the food around isolated %s leader attachments'
        info_description = 'Fisher information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times' \
                           ' around isolated %s leader ant attachments'
        info_result_name = 'fisher_information_' + variable_name

        for suff in ['inside', 'outside']:
            self.exp.load(variable_name % suff)

            ylim_zoom = (0., 1)
            self.__compute_fisher_information_around_leader_attachments(
                self.__compute_fisher_info, variable_name % suff, info_result_name % suff, info_label % suff,
                info_description % suff, ylim_zoom, redo)

    def plot_isolated_inside_and_outside_leader_fisher_information(self):
        outside_name = 'fisher_information_mm1s_food_direction_error_around_isolated_outside_leader_attachments'
        inside_name = 'fisher_information_mm1s_food_direction_error_around_isolated_inside_leader_attachments'
        self.exp.load([outside_name, inside_name])
        result_name = 'fisher_information_isolated_outside_inside_leader_attachment'

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(outside_name))
        fig, ax = plotter.create_plot()
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='Time (s)', ylabel=r'Fisher information (rad$^{-2}$)',
            title='', label='outside', c='red', bar_width=2.7)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(inside_name))
        plotter.plot_with_error(
            preplot=(fig, ax), xlabel='Time (s)', ylabel=r'Fisher information (rad$^{-2}$)',
            title='', label='inside', c='navy')

        ax.set_xlim(-2, 8)
        ax.set_ylim(0., 1.6)
        plotter.draw_vertical_line(ax, label='attachment')

        m = []
        m += list(self.exp.get_df(outside_name).loc[-2:0.5, 'info'][:-1])
        m += list(self.exp.get_df(inside_name).loc[-2:0.5, 'info'][:-1])
        y0 = np.around(np.mean(m), 2)

        plotter.draw_horizontal_line(ax, val=y0, c='w', label='y= %.2f' % y0)
        plotter.draw_legend(ax)

        plotter.save(fig, name=result_name)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(outside_name))
        plotter.column_name = 'info'
        fig, ax = plotter.create_plot()

        plotter.plot(
            fct_y=lambda x: 1 / x,
            preplot=(fig, ax), xlabel='Time (s)', ylabel='Variance', title='', label='outside', c='red')

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(inside_name))
        plotter.column_name = 'info'
        plotter.plot(
            fct_y=lambda x: 1 / x,
            preplot=(fig, ax), xlabel='Time (s)', ylabel='Variance',
            title='', label='inside', c='navy')

        ax.set_xlim(-2, 8)
        # ax.set_ylim(0.3, .8)
        plotter.draw_vertical_line(ax, label='attachment')
        plotter.draw_legend(ax)

        plotter.save(fig, name=result_name+'_var')

    def compute_mm1s_food_direction_error_around_isolated_leading_attachments_around30s(self):
        attachment_name = 'isolated_%s_leading_attachment_intervals'
        variable_name = 'mm1s_food_direction_error'

        result_label = 'Food direction error around isolated %s leader attachments'
        result_description = 'Food direction error smoothed with a moving mean of window 1s for times' \
                             ' before and after an ant coming from %s ant attached to the food and has' \
                             ' an influence on it and that no other leading attachment occurred 2s after and before'

        result_name = variable_name + '_around_isolated_%s_leader_attachments'
        time_min = 40
        time_max = 200
        for suff in ['outside', 'inside']:
            self.exp.load(attachment_name % suff)

            self.__gather_exp_frame_indexed_around_leader_attachments(
                variable_name, attachment_name % suff, result_name % suff + '_before30s',
                result_label % suff, result_description % suff,
                time_min=0, time_max=time_min*100)

            self.__gather_exp_frame_indexed_around_leader_attachments(
                variable_name, attachment_name % suff, result_name % suff + '_after30s',
                result_label % suff, result_description % suff,
                time_min=time_min*100, time_max=time_max*100)

            self.exp.remove_object(attachment_name % suff)
        self.exp.remove_object(variable_name)

    def compute_fisher_information_mm1s_food_direction_error_around_isolated_leader_attachments_around30s(
            self, redo=False):

        variable_name = 'mm1s_food_direction_error_around_isolated_%s_leader_attachments'
        info_label = 'Fisher information of the food around isolated %s leader attachments'
        info_description = 'Fisher information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times' \
                           ' around isolated %s leader ant attachments'
        info_result_name = 'fisher_information_' + variable_name

        for suff in ['inside', 'outside']:
            self.exp.load(variable_name % suff + '_before30s')
            self.exp.load(variable_name % suff + '_after30s')

            self.__compute_fisher_information_around_leader_attachments(
                self.__compute_fisher_info, variable_name % suff + '_before30s', info_result_name % suff + '_before30s',
                info_label % suff, info_description % suff, None, redo, False)

            self.__compute_fisher_information_around_leader_attachments(
                self.__compute_fisher_info, variable_name % suff + '_after30s', info_result_name % suff + '_after30s',
                info_label % suff, info_description % suff, None, redo, False)

            plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name % suff + '_before30s'))
            fig, ax = plotter.plot_with_error(xlabel='time (s)', title=suff, c='k', draw_lims=True, label='before')

            plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name % suff + '_after30s'))
            plotter.plot_with_error(
                preplot=(fig, ax), xlabel='time (s)', title=suff,  c='w', draw_lims=True, label='after')

            ax.axvline(0, ls='--', c='k')
            ax.set_xlim(-2, 8)
            ax.set_ylim(0, 1)
            plotter.draw_legend(ax)
            plotter.save(fig)
            self.exp.remove_object(info_result_name % suff + '_before30s')
            self.exp.remove_object(info_result_name % suff + '_after30s')

    def compute_w10s_fisher_information_mm1s_food_direction_error(self, redo=False, redo_hist=False):

        name = 'mm1s_food_direction_error'
        w = 10
        result_name = 'w%ss_fisher_information_%s' % (w, name)

        bins = 'fd'
        label = 'Fisher information of the object'
        description = 'Fisher information of the object' \
                      ' computed as the inverse of the variance of %s during a time window of %ss' % (name, w)
        if redo:
            self.exp.load([name, 'fps'])

            self.exp.add_copy(old_name=name, new_name=result_name, category=self.category,
                              label=label, description=description)

            df = self.exp.get_df(name).copy().dropna()
            df = df.rolling(window=w*100, center=True, min_periods=100).apply(
                compute_fisher_information_uniform_von_mises)
            # df = 1/df
            df = df.reindex(self.exp.get_index(result_name))
            self.exp.change_df(result_name, df)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter = Plotter(self.exp.root, self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(normed=True, xscale='log', yscale='log')
        plotter.save(fig)

        surv_name = self.compute_surv(name=result_name, redo=redo, redo_hist=redo_hist)
        plotter = Plotter(self.exp.root, self.exp.get_data_object(surv_name))
        fig, ax = plotter.plot(yscale='log')
        ax.set_xticks(range(0, 230, 20))
        ax.grid()
        plotter.save(fig)

        fig, ax = plotter.plot(yscale='log', marker=None)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.grid(which='both')
        ax.set_xlim(0, 10)
        ax.set_ylim(0.05, 1)
        plotter.save(fig, suffix=2)

    def compute_mm1s_food_direction_error_around_leading_attachments2(self):
        variable_name = 'mm1s_food_direction_error'
        discrim_name = 'w10s_fisher_information_mm1s_food_direction_error'
        attachment_name = '%s_leading_attachment_intervals'
        result_name = variable_name + '_around_%s_leading_attachments2'

        result_label = 'Fisher information of the food direction error around %s leading attachments'
        result_description = 'Fisher information of the food direction error' \
                             ' before and after an ant coming from %s ant attached to the food and has an influence'
        for suff in ['outside', 'inside']:

            self.__gather_exp_frame_indexed_around_attachments_with_discrimination(
                variable_name, discrim_name, attachment_name % suff, result_name % suff,
                result_label % suff, result_description % suff)

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

    def compute_fisher_information_mm1s_food_direction_error_around_leading_attachments2(self, redo):

        variable_name = 'mm1s_food_direction_error_around_%s_leading_attachments2'
        info_result_name = 'fisher_information_' + variable_name
        hist_result_name = 'histograms_' + variable_name

        hist_label = 'Histograms of the food direction error around %s leader attachments'
        hist_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                           'which are times around %s leader ant attachments'

        info_label = 'Information of the food around %s leader attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around %s leader ant attachments'

        fi_intervals = [0, 0.15, 0.3, 0.45, 0.65, 1]

        for suff in ['outside', 'inside']:
            self.__compute_mean_around_attachments2(
                self.__compute_fisher_info2, fi_intervals,
                variable_name % suff, info_result_name % suff, hist_result_name % suff,
                info_label % suff, info_description % suff,
                hist_label % suff, hist_description % suff, redo)

        temp_name = 'temp_name'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(hist_result_name % suff))
        fig, ax = plotter.create_plot(
            figsize=(8, 15), nrows=5, ncols=2)
        title = r'%s: fisher info $\in$[%.2f, %.2f]'
        for k, fi0 in enumerate(fi_intervals[:-1]):
            fi1 = fi_intervals[k+1]
            # if k < 5:
            #     j = int(k / 5)
            #     i = k - j * 5
            # else:
            #     j = int((k-5) / 5)+2
            #     i = (k-5)-(j-2)*5
            # print(k, i, j)

            suff = 'outside'
            df = self.exp.get_df(hist_result_name % suff).loc[fi0, :]
            self.exp.add_new_dataset_from_df(df, temp_name, category=self.category, replace=True)
            plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(temp_name))
            plotter.plot(
                preplot=(fig, ax[k, 0]), title=title % (suff, fi0, fi1),
                xlabel='Path Efficiency', ylabel='PDF', normed=True)
            ax[k, 0].set_xlim(0, np.pi)
            ax[k, 0].set_ylim(0, 1)

            suff = 'inside'
            df = self.exp.get_df(hist_result_name % suff).loc[fi0, :]
            self.exp.add_new_dataset_from_df(df, temp_name, category=self.category, replace=True)
            plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(temp_name))
            plotter.plot(
                preplot=(fig, ax[k, 1]), title=title % (suff, fi0, fi1),
                xlabel='Path Efficiency', ylabel='PDF', normed=True)
            ax[k, 1].set_xlim(0, np.pi)
            ax[k, 1].set_ylim(0, 1)
        plotter.save(fig, name=hist_result_name % '')

        temp_name = 'temp_name'
        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name % suff))
        ncols = 5
        fig, ax = plotter.create_plot(
            figsize=(15, 4), ncols=5, left=0.04, top=0.92, bottom=0.05)
        title = r'fisher info $\in$[%.2f, %.2f]'
        for k, fi0 in enumerate(fi_intervals[:-1]):
            fi1 = fi_intervals[k+1]
            # i = int(k / ncols)
            # j = k - i * ncols
            suff = 'outside'
            df = self.exp.get_df(info_result_name % suff).loc[fi0, :]
            self.exp.add_new_dataset_from_df(df, temp_name, category=self.category, replace=True)
            plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(temp_name))
            plotter.plot_with_error(
                preplot=(fig, ax[k]), xlabel='time (s)',
                ylabel='Mean path efficiency', title=title % (fi0, fi1), c='red', label=suff)

            suff = 'inside'
            df = self.exp.get_df(info_result_name % suff).loc[fi0, :]
            self.exp.add_new_dataset_from_df(df, temp_name, category=self.category, replace=True)
            plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(temp_name))
            plotter.plot_with_error(
                preplot=(fig, ax[k]), xlabel='time (s)',
                ylabel='Mean path efficiency', title=title % (fi0, fi1), c='navy', label=suff)

            plotter.draw_vertical_line(ax[k])
            plotter.draw_horizontal_line(ax[k], fi0, c='w')
            plotter.draw_horizontal_line(ax[k], fi_intervals[k+1], c='w')
            ax[k].set_xlim(-2, 8)
            # ax[k].set_ylim(0, 5)
        plotter.save(fig, name=info_result_name % '')

    def __compute_fisher_info2(self, time_intervals, fi_intervals, variable_name, result_name):
        for j in range(len(fi_intervals) - 1):
            r = 0
            fi0 = fi_intervals[j]
            fi1 = fi_intervals[j + 1]
            for t in time_intervals[:-1]:
                values = self.exp.get_df(variable_name).loc[pd.IndexSlice[:, :, fi0:fi1], str(t)]
                r += len(values)

                # fisher_info = round(1 / np.var(values), 3)
                # fisher_info = round(compute_fisher_information_uniform_von_mises(values), 3)

                fisher_info, q, kappa, cov_mat = compute_fisher_information_uniform_von_mises(
                    values, get_fit_quality=True)
                q_err, kappa_err = 1.96 * np.sqrt(np.diagonal(cov_mat))

                q1 = max(q - q_err, 0)
                q2 = min(q + q_err, 1)
                kappa1 = max(kappa - kappa_err, 0)
                kappa2 = kappa + kappa_err
                fi1 = compute_fisher_information_uniform_von_mises_fix(q1, kappa1)
                fi2 = compute_fisher_information_uniform_von_mises_fix(q2, kappa2)

                self.exp.get_df(result_name).loc[t, 'info'] = fisher_info
                self.exp.get_df(result_name).loc[t, 'CI95_2'] = fi2 - fisher_info
                self.exp.get_df(result_name).loc[t, 'CI95_1'] = fisher_info - fi1

                # lg = len(values)
                # fisher_list = []
                #
                # i = 0
                # while i < 500:
                #     try:
                #         idx = np.random.randint(0, lg, lg)
                #         val = values[idx]
                #         fi = compute_fisher_information_uniform_von_mises(val)
                #         # fisher_list.append(1 / np.var(val))
                #         fisher_list.append(fi)
                #         i += 1
                #     except RuntimeError:
                #         pass
                #
                # self.exp.get_df(result_name).loc[fi0, t]['info'] = fisher_info
                # ci95 = round(fisher_info - np.percentile(fisher_list, 2.5), 3)
                # self.exp.get_df(result_name).loc[fi0, t]['CI95_1'] = ci95
                # ci95 = round(np.percentile(fisher_list, 97.5) - fisher_info, 3)
                # self.exp.get_df(result_name).loc[fi0, t]['CI95_2'] = ci95

            print(fi0, r / (len(fi_intervals) - 1))

    def __compute_mean_around_attachments2(
            self, fct, fi_intervals, variable_name, info_result_name, hist_result_name,
            info_label, info_description, hist_label, hist_description, redo, list_exps=None):

        t0, t1, dt = -9.5, 9.5, 2.
        time_intervals = np.around(np.arange(t0, t1+1 + dt, dt), 1)

        dtheta = np.pi/10.
        bins = np.arange(0, np.pi+dtheta, dtheta)
        hists_index_values = np.around((bins[1:] + bins[:-1]) / 2., 3)

        index_values = [(fi, h) for fi in fi_intervals[:-1] for h in hists_index_values]

        if redo:
            self.exp.load(variable_name)

            self.exp.add_new_empty_dataset(name=hist_result_name,
                                           index_names=['info', 'food_direction_error'],
                                           column_names=time_intervals[:-1], index_values=index_values,
                                           category=self.category, label=hist_label, description=hist_description,
                                           replace=True)

            for i in range(len(time_intervals)-1):
                t0 = str(time_intervals[i])
                t1 = str(time_intervals[i+1])
                for j in range(len(fi_intervals) - 1):
                    fi0 = fi_intervals[j]
                    fi1 = fi_intervals[j + 1]

                    values = np.abs(self.exp.get_df(variable_name).loc[pd.IndexSlice[:, :, fi0:fi1], t0:t1].dropna())
                    if list_exps is not None:
                        values = values.loc[list_exps, :]
                    values = values.values.ravel()
                    hist = np.histogram(values, bins=bins, normed=False)[0]

                    self.exp.get_df(hist_result_name).loc[pd.IndexSlice[fi0, :], float(t0)] = hist

            t0, t1, dt = -2, 8, 1.
            time_intervals = np.around(np.arange(t0, t1 + 1 + dt, dt), 1)
            index_values = [(fi, t) for fi in fi_intervals[:-1] for t in time_intervals]

            self.exp.add_new_empty_dataset(name=info_result_name, index_names=['fisher_info', 'time'],
                                           column_names=['info', 'CI95_1', 'CI95_2'],
                                           index_values=index_values, category=self.category,
                                           label=info_label, description=info_description, replace=True)

            fct(time_intervals, fi_intervals, variable_name, info_result_name)

            self.exp.remove_object(variable_name)
            self.exp.write(hist_result_name)
            self.exp.write(info_result_name)
        else:
            self.exp.load(info_result_name)
            self.exp.load(hist_result_name)

    def compute_mm1s_food_direction_error_discrim_w10s_food_path_efficiency_around_leading_attachments(self):
        variable_name = 'mm1s_food_direction_error'
        discrim_name = 'w10s_food_path_efficiency'
        attachment_name = '%s_leading_attachment_intervals'

        result_label = 'Fisher information of the food direction error around %s leading attachments'
        result_description = 'Fisher information of the food direction error' \
                             ' before and after an ant coming from %s ant attached to the food and has an influence'
        for suff in ['outside', 'inside']:
            result_name = '%s_discrim_%s_around_%s_leading_attachments' % (variable_name, discrim_name, suff)

            self.__gather_exp_frame_indexed_around_attachments_with_discrimination(
                variable_name, discrim_name, attachment_name % suff, result_name,
                result_label % suff, result_description % suff)

    def compute_fisher_info_mm1s_food_direction_error_discrim_w10s_food_path_efficiency_around_leading_attachments(
            self, redo):

        variable_name = 'mm1s_food_direction_error_discrim_w10s_food_path_efficiency_around_%s_leading_attachments'
        info_result_name = 'fisher_information_' + variable_name
        hist_result_name = 'histograms_' + variable_name

        hist_label = 'Histograms of the food direction error around %s leader attachments'
        hist_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                           'which are times around %s leader ant attachments'

        info_label = 'Information of the food around %s leader attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around %s leader ant attachments'

        dx = 0.1
        a0, a1, da = 0, 1+dx, dx
        eff_intervals = np.around(np.arange(a0, a1, da), 1)

        for suff in ['outside', 'inside']:
            self.__compute_mean_around_attachments2(
                self.__compute_fisher_info2, eff_intervals,
                variable_name % suff, info_result_name % suff, hist_result_name % suff,
                info_label % suff, info_description % suff,
                hist_label % suff, hist_description % suff, redo)

        self._plot_mm1s_food_direction_error_around_attachments(suff, info_result_name, hist_result_name, eff_intervals)

    def _plot_mm1s_food_direction_error_around_attachments(
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
            ax[i, j].set_xlim(0, np.pi)
            ax[i, j].set_title(title % (suff, eff, eff1), color='r')

            suff = 'inside'
            df = self.exp.get_df(hist_result_name % suff).loc[eff, :]
            self.exp.add_new_dataset_from_df(df, temp_name, category=self.category, replace=True)
            plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(temp_name))
            plotter.plot(preplot=(fig, ax[i, j + 1]), xlabel='Path Efficiency', ylabel='PDF', normed=True)
            ax[i, j + 1].set_xlim(0, np.pi)
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
            ax[i, j].set_ylim(0, 2.)

        plotter.save(fig, name=mean_result_name % '')

    def compute_mm1s_food_direction_error_discrim_mm1s_food_direction_error_variation_around_leading_attachments(self):
        variable_name = 'mm1s_food_direction_error'
        discrim_name = 'mm1s_food_direction_error_variation'
        attachment_name = '%s_leading_attachment_intervals'

        result_label = 'Fisher information of the food direction error around %s leading attachments'
        result_description = 'Fisher information of the food direction error' \
                             ' before and after an ant coming from %s ant attached to the food and has an influence'
        for suff in ['outside', 'inside']:
            result_name = '%s_discrim_%s_around_%s_leading_attachments' % (variable_name, discrim_name, suff)

            self.__gather_exp_frame_indexed_around_attachments_with_discrimination(
                variable_name, discrim_name, attachment_name % suff, result_name,
                result_label % suff, result_description % suff)

    def compute_fisher_info_mm1s_food_direction_error_discrim_mm1s_food_error_variation_around_leading_attachments(
            self, redo):

        variable_name =\
            'mm1s_food_direction_error_discrim_mm1s_food_direction_error_variation_around_%s_leading_attachments'
        info_result_name = 'fisher_information_' + variable_name
        hist_result_name = 'histograms_' + variable_name

        hist_label = 'Histograms of the food direction error around %s leader attachments'
        hist_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                           'which are times around %s leader ant attachments'

        info_label = 'Information of the food around %s leader attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around %s leader ant attachments'

        eff_intervals = [0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.15, 1.55, 2.2, np.pi]

        for suff in ['outside', 'inside']:
            self.__compute_mean_around_attachments2(
                self.__compute_fisher_info2, eff_intervals,
                variable_name % suff, info_result_name % suff, hist_result_name % suff,
                info_label % suff, info_description % suff,
                hist_label % suff, hist_description % suff, redo)

        self._plot_mm1s_food_direction_error_around_attachments(suff, info_result_name, hist_result_name, eff_intervals)
