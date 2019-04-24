import numpy as np
import pandas as pd

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_frame_name
from Tools.MiscellaneousTools.ArrayManipulation import get_entropy, get_max_entropy
from Tools.Plotter.ColorObject import ColorObject
from Tools.Plotter.Plotter import Plotter


class AnalyseFoodEntropy(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'FoodEntropy'

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
        variable_name = 'mm' + str(mm) + 's_food_direction_error'

        result_name = variable_name + '_around_outside_attachments'

        result_label = 'Food direction error around outside attachments'
        result_description = 'Food direction error smoothed with a moving mean of window ' + str(mm) + ' s for times' \
                             ' before and after a ant coming from outside ant attached to the food'

        self.__gather_variable_around_attachments(variable_name, attachment_name, result_name, result_label,
                                                  result_description)

    def compute_mm1s_food_direction_error_around_non_outside_attachments(self):
        mm = 1
        attachment_name = 'non_outside_ant_carrying_intervals'
        variable_name = 'mm' + str(mm) + 's_food_direction_error'

        result_name = variable_name + '_around_non_outside_attachments'

        result_label = 'Food direction error around non outside attachments'
        result_description = 'Food direction error smoothed with a moving mean of window ' + str(mm) + ' s for times' \
                             ' before and after a ant coming from non outside ant attached to the food'

        self.__gather_variable_around_attachments(variable_name, attachment_name, result_name, result_label,
                                                  result_description)

    def __gather_variable_around_attachments(self, variable_name, attachment_name, result_name, result_label,
                                             result_description):

        last_frame_name = 'food_exit_frames'
        self.exp.load([attachment_name, variable_name, last_frame_name, 'fps'])

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

            attachment_frames = self.exp.get_df(attachment_name).loc[id_exp, :]
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

        ylim_zoom = (0.4, 0.6)
        self.__compute_information_around_attachments(variable_name, hists_result_name, info_result_name, hists_label,
                                                      hists_description, info_label, info_description, ylim_zoom,
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

        info_label = 'Information of the food around outside attachments'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around non outside ant attachments'

        ylim_zoom = (0.1, 0.2)
        self.__compute_information_around_attachments(variable_name, hists_result_name, info_result_name, hists_label,
                                                      hists_description, info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

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

        ylim_zoom = (0.2, 0.3)
        self.__compute_information_around_attachments(variable_name, hists_result_name, info_result_name, hists_label,
                                                      hists_description, info_label, info_description, ylim_zoom,
                                                      redo, redo_info, redo_plot_hist)

    def __compute_information_around_attachments(self, variable_name, hists_result_name, info_result_name, hists_label,
                                                 hists_description, info_label, info_description, ylim_zoom,
                                                 redo, redo_info, redo_plot_hist):
        t0, t1, dt = -60, 60, 0.5
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 1)
        dtheta = np.pi / 12.
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

            self.exp.write(hists_result_name)

        else:
            self.exp.load(hists_result_name)

        if redo or redo_info:
            time_intervals = self.exp.get_df(hists_result_name).columns

            self.exp.add_new_empty_dataset(name=info_result_name, index_names='time', column_names='info',
                                           index_values=time_intervals, category=self.category,
                                           label=info_label, description=info_description)

            max_entropy = get_max_entropy(hists_index_values)

            for t in time_intervals:
                hist = self.exp.get_df(hists_result_name)[t]
                entropy = get_entropy(hist)

                self.exp.get_df(info_result_name).loc[t] = np.around(max_entropy - entropy, 2)

            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)

        if redo or redo_info or redo_plot_hist:
            for t in time_intervals:
                plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(hists_result_name), column_name=t)
                fig, ax = plotter.plot(xlabel='Food direction error', ylabel='Probability')
                ax.set_ylim((0, 0.25))
                plotter.save(fig, sub_folder=hists_result_name, name=t)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.create_plot(figsize=(5, 8), nrows=2)

        plotter.plot(preplot=(fig, ax[0]), xlabel='time (s)', ylabel='Information (bit)', title='')
        ax[0].axvline(0, ls='--', c='k')

        plotter.plot(preplot=(fig, ax[1]), title='')
        ax[1].axvline(0, ls='--', c='k')
        ax[1].set_xlim((-2, 8))
        ax[1].set_ylim(ylim_zoom)

        plotter.save(fig)

    def compute_information_mm1s_food_direction_error_around_first_outside_attachments(
            self, redo=False, redo_info=False, redo_plot_hist=False):

        variable_name = 'mm1s_food_direction_error_around_outside_attachments'
        self.exp.load([variable_name])

        hists_result_name = 'histograms_mm1s_food_direction_error_around_first_outside_attachments'
        info_result_name = 'information_mm1s_food_direction_error_around_first_outside_attachments'

        hists_label = 'Histograms of the food direction error around first outside attachments'
        hists_description = 'Histograms of the food direction error for each time t in time_intervals, ' \
                            'which are times around first outside ant attachments'

        info_label = 'Information of the food around outside first outside attachments'
        info_description = 'Information of the food (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around first outside ant attachments'

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

            max_entropy = get_max_entropy(bins2)

            for (th, t) in self.exp.get_index(hists_result_name):
                hist = self.exp.get_df(hists_result_name).loc[(th, t)]
                entropy = get_entropy(hist)

                self.exp.get_df(info_result_name).loc[t, th] = np.around(max_entropy - entropy, 2)

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
                    df = pd.DataFrame(data=np.array(df), index=bins2, columns=['temp'])
                    self.exp.add_new_dataset_from_df(df=df, name='temp', category=self.category, replace=True)

                    plotter = Plotter(self.exp.root, obj=self.exp.get_data_object('temp'))
                    fig, ax = plotter.plot(preplot=(fig, ax), xlabel='Food direction error', ylabel='Probability',
                                           label=str(th)+'th', c=colors[str(th)])

                ax.set_title(str(t)+' (s)')
                ax.set_ylim((0, 0.4))
                ax.legend()
                plotter.save(fig, sub_folder=hists_result_name, name=t)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.create_plot(figsize=(5, 8), nrows=2)

        plotter.plot_smooth(window=150, preplot=(fig, ax[0]), marker='',
                            xlabel='time (s)', ylabel='Information (bit)', title='')
        ax[0].axvline(0, ls='--', c='k')

        plotter.plot_smooth(window=150, preplot=(fig, ax[1]), title='', marker='',)
        ax[1].axvline(0, ls='--', c='k')
        ax[1].set_xlim((-2, 8))
        ax[1].set_ylim((.35, .75))

        plotter.save(fig)

    def __extract_10first_attachments(self, first_attach_name, variable_name):
        first_attach_index = [(id_exp, th) for id_exp in self.exp.id_exp_list for th in range(1, 11)]
        self.exp.add_new_empty_dataset(name=first_attach_name, index_names=[id_exp_name, 'rank'],
                                       column_names=self.exp.get_df(variable_name).columns,
                                       index_values=first_attach_index)

        def get_10first_attachment(df):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            print(id_exp)
            self.exp.get_df(first_attach_name).loc[id_exp, :] = np.array(df.iloc[:10, :])

        self.exp.groupby(variable_name, id_exp_name, get_10first_attachment)

    def compute_information_mm1s_food_direction_error_around_the_first_outside_attachment(self, redo=False):

        variable_name = 'mm1s_food_direction_error'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        self.exp.load([variable_name, first_attachment_name, 'fps'])

        info_result_name = 'information_mm1s_food_direction_error_around_the_first_outside_attachment'

        info_label = 'Information of the food around outside the first outside attachment'
        info_description = 'Information of the food  (max entropy - entropy of the food direction error)' \
                           ' for each time t in time_intervals, which are times around the first outside ant attachment'

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

            max_entropy = get_max_entropy(bins2)
            for t in time_intervals:
                values = self.exp.get_df('temp').loc[t].dropna()
                hist = np.histogram(values, bins=bins, density=False)[0]
                s = float(np.sum(hist))
                hist = hist / s
                entropy = get_entropy(hist)

                self.exp.get_df(info_result_name).loc[t] = np.around(max_entropy - entropy, 2)

                if not(np.isnan(entropy)) and t in time_intervals_to_plot:
                    df = pd.DataFrame(data=hist, index=bins2, columns=['hist'])
                    self.exp.add_new_dataset_from_df(df=df, name='hist', category=self.category, replace=True)

                    plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('hist'))
                    fig, ax = plotter.plot(xlabel='Food direction error', ylabel='Probability')

                    plotter.save(fig, sub_folder=info_result_name, name=t)

            self.exp.write(info_result_name)

        else:
            self.exp.load(info_result_name)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(info_result_name))
        fig, ax = plotter.create_plot(figsize=(5, 8), nrows=2)

        plotter.plot(preplot=(fig, ax[0]), xlabel='time (s)', ylabel='Information (bit)', title='')
        ax[0].axvline(0, ls='--', c='k')

        plotter.plot(preplot=(fig, ax[1]), title='')
        ax[1].axvline(0, ls='--', c='k')
        ax[1].set_xlim((-2, 8))
        ax[1].set_ylim((0, 1))

        plotter.save(fig)
