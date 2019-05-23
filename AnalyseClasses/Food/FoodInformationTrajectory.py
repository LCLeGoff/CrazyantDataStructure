import numpy as np
import pandas as pd

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_frame_name
from Tools.Plotter.Plotter import Plotter


class AnalyseFoodInformationTrajectory(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'FoodInformationTrajectory'
        self.exp.data_manager.create_new_category(self.category)

    def compute_w10s_food_direction_error_vs_path_efficiency(self):
        time = 10
        confidence_name = 'w'+str(time)+'s_food_path_efficiency'
        veracity_name = 'mm'+str(time)+'s_food_direction_error'
        result_name = 'w'+str(time)+'s_food_direction_error_vs_path_efficiency'

        xlabel = 'Food path efficiency'
        ylabel = 'Direction error'

        self.__compute_indiv_info_traj(result_name, confidence_name, veracity_name, xlabel, ylabel)

    def compute_w30s_food_direction_error_vs_path_efficiency(self):
        time = 30
        confidence_name = 'w'+str(time)+'s_food_path_efficiency'
        veracity_name = 'mm'+str(time)+'s_food_direction_error'
        result_name = 'w'+str(time)+'s_food_direction_error_vs_path_efficiency'

        xlabel = 'Food path efficiency'
        ylabel = 'Direction error'

        self.__compute_indiv_info_traj(result_name, confidence_name, veracity_name, xlabel, ylabel)

    def __compute_indiv_info_traj(self, result_name, confidence_name, veracity_name, xlabel, ylabel):
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        last_frame_name = 'food_exit_frames'

        self.exp.load([confidence_name, veracity_name, first_attachment_name, last_frame_name])

        def plot2d(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            frame_attach = self.exp.get_value(first_attachment_name, id_exp)
            last_frame = self.exp.get_value(last_frame_name, id_exp)
            print(id_exp)

            confidence_df = df.loc[id_exp, :]
            error_df = self.exp.get_df(veracity_name).loc[id_exp, :]
            error_df = 1-error_df.abs()/np.pi
            confidence_df = confidence_df.loc[:last_frame]
            error_df = error_df.loc[:last_frame]

            df_to_plot = confidence_df.join(error_df)
            df_to_plot.dropna(inplace=True)
            df_to_plot = df_to_plot.set_index(confidence_name)
            self.exp.add_new_dataset_from_df(df=df_to_plot, name='to_plot', category=self.category, replace=True)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('to_plot'))
            fig, ax = plotter.create_plot()
            plotter.plot(preplot=(fig, ax), xlabel=xlabel, ylabel=ylabel, marker='', title=id_exp)

            self.exp.add_new_dataset_from_df(df=df_to_plot.iloc[0:1],
                                             name='to_plot', category=self.category, replace=True)
            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('to_plot'))
            plotter.plot(preplot=(fig, ax), xlabel=xlabel, ylabel=ylabel, ls='', title=id_exp, c='r')

            df_to_plot = confidence_df.loc[frame_attach:].join(error_df.loc[frame_attach:])
            df_to_plot.dropna(inplace=True)
            df_to_plot = df_to_plot.set_index(confidence_name)
            self.exp.add_new_dataset_from_df(df=df_to_plot, name='to_plot', category=self.category, replace=True)
            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('to_plot'))
            plotter.plot(preplot=(fig, ax), xlabel=xlabel, ylabel=ylabel, marker='', title=id_exp, c='b')
            self.exp.add_new_dataset_from_df(df=df_to_plot.iloc[-2:-1],
                                             name='to_plot', category=self.category, replace=True)
            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('to_plot'))
            plotter.plot(preplot=(fig, ax), xlabel=xlabel, ylabel=ylabel, ls='', title=id_exp, c='g')

            ax.set_xlim((0, 1))
            ax.set_ylim((0, 1))
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.set_xticks(np.arange(0, 1.1, 0.2))
            ax.grid()
            plotter.save(fig, name=id_exp, sub_folder=result_name)

        self.exp.groupby(confidence_name, id_exp_name, plot2d)

    def w10s_food_direction_error_vs_path_efficiency_hist2d(self, redo=False):
        time = 10
        self.__plot_heatmap_info_traj(time, redo)

    def w30s_food_direction_error_vs_path_efficiency_hist2d(self, redo=False):
        time = 30
        self.__plot_heatmap_info_traj(time, redo)

    def __plot_heatmap_info_traj(self, time, redo):
        w = 10

        result_name_all = 'w' + str(time) + 's_food_direction_error_vs_path_efficiency_hist2d'
        result_name_start = 'w' + str(time) + 's_food_direction_error_vs_path_efficiency_hist2d_start'
        result_name_end = 'w' + str(time) + 's_food_direction_error_vs_path_efficiency_hist2d_end'

        result_names = [result_name_all, result_name_start, result_name_end]

        dc = 0.1
        x_intervals = np.arange(0, 1 + dc, dc)
        dv = 0.1
        y_intervals = np.arange(0, 1 + dv, dv)

        name_x = 'w' + str(time) + 's_food_path_efficiency'
        name_y = 'mm' + str(time) + 's_food_direction_error'
        name_attachments = 'outside_ant_attachment_frames'
        name_last_frame = 'food_exit_frames'

        label = '2D histogram of food path efficiency vs food direction error'
        description = 'Two-D histogram of the 2 variables food path efficiency and food direction error'

        if redo:

            self.exp.load([name_x, name_y, name_attachments, name_last_frame, 'fps'])
            self.exp.operation(name_y, lambda df: 1-(np.abs(df) / np.pi))

            x = self.exp.get_df(name_x)[name_x]
            y = self.exp.get_df(name_y)[name_y]

            self.__compute_hist2d_info_traj_all_times(
                x, result_name_all, y, x_intervals, y_intervals, label, description)

            self.__compute_hist2d_info_traj_at_start(
                x, y, result_name_start, w, x_intervals, y_intervals, label, description)

            self.__compute_hist2d_info_traj_at_end(
                x, y, name_last_frame, result_name_end, w, x_intervals, y_intervals, label, description)

            self.exp.write(result_names)
            self.exp.remove_object(name_y)
        else:
            self.exp.load(result_names)

        for result_name in result_names:
            plotter = Plotter(self.exp.root, self.exp.get_data_object(result_name))
            fig, ax, cbar = plotter.plot_heatmap(xlabel='Confidence', ylabel='Veracity', cbar_label='Proportion')
            plotter.save(fig)

    def __compute_hist2d_info_traj_at_start(self, x, y, result_name, w, x_intervals, y_intervals, label, description):

        label_start = label + ' for the first ' + str(w) + 's'
        description_start = description + ' for the first ' + str(w) + 's'

        def func_start(df, id_exp, dframe):
            frame = df.loc[id_exp, :].index.get_level_values(id_frame_name)[0]
            return list(df.loc[id_exp, frame:frame + dframe].index.get_level_values(id_frame_name))

        self.__compute_info_traj_hist2d_at_specific_times(
            result_name, x, y, x_intervals, y_intervals, func_start, w, label_start, description_start)

    def __compute_hist2d_info_traj_at_end(
            self, x, y, name_variable, result_name, w, x_intervals, y_intervals, label, description):

        label_end = label + ' for the last ' + str(w) + 's'
        description_end = description + ' for the last ' + str(w) + 's'

        def func_end(df, id_exp, dframe):
            frame = self.exp.get_value(name_variable, id_exp)
            return list(df.loc[id_exp, frame - dframe:frame].index.get_level_values(id_frame_name))

        self.__compute_info_traj_hist2d_at_specific_times(
            result_name, x, y, x_intervals, y_intervals, func_end, w, label_end, description_end)

    def __compute_hist2d_info_traj_all_times(self, x, result_name_all, y, x_intervals, y_intervals, label, description):
        self.__compute_hist2d(x, y, x_intervals, y_intervals, result_name_all, label, description)

    def __compute_info_traj_hist2d_at_specific_times(
            self, result_name, x, y, x_intervals, y_intervals, func, w, label, description):

        indexes = []
        for id_exp in self.exp.id_exp_list:
            fps = self.exp.get_value('fps', id_exp)
            dframe = w * fps
            frames = func(x, id_exp, dframe)
            indexes += [(id_exp, frame) for frame in frames]
        x2 = x.reindex(indexes)
        y2 = y.reindex(indexes)
        self.__compute_hist2d(x2, y2, x_intervals, y_intervals, result_name, label, description)

    def __compute_hist2d(self, x, y, x_intervals, y_intervals, result_name, label, description):
        h, xedges, yedges = np.histogram2d(x=x, y=y, bins=[x_intervals, y_intervals])
        res = np.zeros((h.shape[0], h.shape[1] + 1))
        res[:, 1:] = h
        res[:, 0] = xedges[:-1]
        self.exp.add_new_dataset_from_array(array=res, name=result_name, index_names='confidence',
                                            column_names=yedges[:-1], category=self.category, label=label,
                                            description=description)
        self.exp.get_data_object(result_name).df = self.exp.get_df(result_name).astype(int, inplace=True)

    def w10s_food_direction_error_vs_path_efficiency_hist2d_around_first_outside_attachment(self, redo=False):
        time = 10
        self.__plot_heatmap_info_traj_around_first_outside_attachment(time, redo)

    def w30s_food_direction_error_vs_path_efficiency_hist2d_around_first_outside_attachment(self, redo=False):
        time = 30
        self.__plot_heatmap_info_traj_around_first_outside_attachment(time, redo)

    def __plot_heatmap_info_traj_around_first_outside_attachment(self, time, redo):
        result_name = 'w' + str(time) + 's_food_direction_error_vs_path_efficiency_hist2d_w'
        label = '2D histogram of food path efficiency vs food direction error'
        description = 'Two-D histogram of the 2 variables food path efficiency and food direction error'

        dt_before_attach = [-120, -30, -10, -5]
        dt_after_attach = [0, 5, 10, 20, 30, 600]
        dt_attach = dt_before_attach+dt_after_attach

        result_names = []
        labels = []
        descriptions = []

        for i in range(len(dt_before_attach)-1):
            t0 = dt_before_attach[i]
            t1 = dt_before_attach[i+1]
            result_names.append(result_name+str(-t0)+'_'+str(-t1)+'s_before_first_outside_attachment')
            labels.append(label+' between '+str(t0)+' and '+str(t1)+' s')
            descriptions.append(description+' between '+str(t0)+' and '+str(t1)+' s')

        t0 = dt_before_attach[-1]
        t1 = dt_after_attach[0]
        result_names.append(result_name+str(-t0)+'_'+str(t1)+'s_before_first_outside_attachment')
        labels.append(label+' between '+str(t0)+' and '+str(t1)+' s')
        descriptions.append(description+' between '+str(t0)+' and '+str(t1)+' s')

        for i in range(len(dt_after_attach)-1):
            t0 = dt_after_attach[i]
            t1 = dt_after_attach[i+1]
            result_names.append(result_name+str(t0)+'_'+str(t1)+'s_after_first_outside_attachment')
            labels.append(label+' between '+str(t0)+' and '+str(t1)+' s')
            descriptions.append(description+' between '+str(t0)+' and '+str(t1)+' s')

        dc = 0.1
        x_intervals = np.arange(0, 1 + dc, dc)

        dv = 0.1
        y_intervals = np.arange(0, 1 + dv, dv)

        if redo:

            name_x = 'w' + str(time) + 's_food_path_efficiency'
            name_y = 'mm' + str(time) + 's_food_direction_error'
            name_attachments = 'outside_ant_attachment_frames'
            name_last_frame = 'food_exit_frames'

            self.exp.load([name_x, name_y, name_attachments, name_last_frame, 'fps'])
            self.exp.operation(name_y, lambda df: 1 - (np.abs(df) / np.pi))

            x = self.exp.get_df(name_x)[name_x]
            y = self.exp.get_df(name_y)[name_y]

            for i in range(len(dt_attach)-1):
                print(i)
                indexes = []

                for id_exp in self.exp.id_exp_list:
                    fps = self.exp.get_value('fps', id_exp)
                    last_frame = self.exp.get_value(name_last_frame, id_exp)
                    frame_attach = self.exp.get_value(name_attachments, (id_exp, 1))

                    f0 = frame_attach+dt_attach[i]*fps
                    f1 = min(frame_attach+dt_attach[i+1]*fps, last_frame)

                    frames = list(x.loc[id_exp, f0:f1].index.get_level_values(id_frame_name))
                    indexes += [(id_exp, frame) for frame in frames]

                x2 = x.reindex(indexes)
                y2 = y.reindex(indexes)
                self.__compute_hist2d(x2, y2, x_intervals, y_intervals, result_names[i], labels[i], descriptions[i])

            self.exp.remove_object(name_y)
            self.exp.write(result_names)
        else:
            self.exp.load(result_names)

        plotter = Plotter(self.exp.root, self.exp.get_data_object(result_names[0]))
        nr = int(round(np.sqrt(len(dt_attach))))
        nc = int(round(len(dt_attach)/nr))
        fig, ax = plotter.create_plot(figsize=(10, 10), nrows=nr, ncols=nc, left=0.05)
        for i in range(len(dt_attach)-1):
            name = result_names[i]
            t0 = dt_attach[i]
            t1 = dt_attach[i+1]
            title = '['+str(t0)+', '+str(t1)+'] s'
            r = int(i/nc)
            c = i % nc

            plotter = Plotter(self.exp.root, self.exp.get_data_object(name))
            plotter.plot_heatmap(
                preplot=(fig, ax[r, c]), xlabel='', ylabel='', cbar_label='', title=title, normed=True)
        ax[1, 0].set_ylabel('Veracity')
        ax[-1, 1].set_xlabel('Confidence')
        plotter.save(fig, name=result_name)

    def w10s_food_direction_error_vs_path_efficiency_hist2d_around_first_outside_attachment_norm_time(self, redo=False):
        time = 10
        self.__plot_heatmap_info_traj_around_first_outside_attachment_norm_time(time, redo)

    def w30s_food_direction_error_vs_path_efficiency_hist2d_around_first_outside_attachment_norm_time(self, redo=False):
        time = 30
        self.__plot_heatmap_info_traj_around_first_outside_attachment_norm_time(time, redo)

    def __plot_heatmap_info_traj_around_first_outside_attachment_norm_time(self, time, redo):
        result_name = 'w' + str(time) + 's_food_direction_error_vs_path_efficiency_hist2d_norm_time_w'
        label = '2D histogram of food path efficiency vs food direction error (time normalized)'
        description = 'Two-D histogram of the 2 variables food path efficiency and food direction error, ' \
                      'the time is normalized: from the first attachment to the last frame,' \
                      ' we count one unit of time'

        dt_before_attach = [-1.0]+list(np.around(np.arange(-0.4, 0, 0.1), 1))
        dt_after_attach = list(np.around(np.arange(0, 0.7, 0.1), 1))+[1.]
        dt_attach = dt_before_attach+dt_after_attach

        result_names = []
        labels = []
        descriptions = []

        for i in range(len(dt_before_attach)-1):
            t0 = dt_before_attach[i]
            t1 = dt_before_attach[i+1]
            result_names.append(result_name+str(-t0)+'_'+str(-t1)+'s_before_first_outside_attachment')
            labels.append(label+' between '+str(t0)+' and '+str(t1))
            descriptions.append(description+' between '+str(t0)+' and '+str(t1))

        t0 = dt_before_attach[-1]
        t1 = dt_after_attach[0]
        result_names.append(result_name+str(-t0)+'_'+str(t1)+'s_before_first_outside_attachment')
        labels.append(label+' between '+str(t0)+' and '+str(t1))
        descriptions.append(description+' between '+str(t0)+' and '+str(t1))

        for i in range(len(dt_after_attach)-1):
            t0 = dt_after_attach[i]
            t1 = dt_after_attach[i+1]
            result_names.append(result_name+str(t0)+'_'+str(t1)+'s_after_first_outside_attachment')
            labels.append(label+' between '+str(t0)+' and '+str(t1))
            descriptions.append(description+' between '+str(t0)+' and '+str(t1))

        dc = 0.1
        x_intervals = np.arange(0, 1 + dc, dc)

        dv = 0.1
        y_intervals = np.arange(0, 1 + dv, dv)

        if redo:

            name_x = 'w' + str(time) + 's_food_path_efficiency'
            name_y = 'mm' + str(time) + 's_food_direction_error'
            name_attachments = 'outside_ant_attachment_frames'
            name_last_frame = 'food_exit_frames'

            self.exp.load([name_x, name_y, name_attachments, name_last_frame])
            self.exp.operation(name_y, lambda df: 1 - (np.abs(df) / np.pi))

            x = self.exp.get_df(name_x)[name_x]
            y = self.exp.get_df(name_y)[name_y]

            for i in range(len(dt_attach)-1):
                print(i)
                indexes = []

                norm_f0 = dt_attach[i]
                norm_f1 = dt_attach[i+1]

                for id_exp in self.exp.id_exp_list:
                    last_frame = self.exp.get_value(name_last_frame, id_exp)
                    frame_attach = self.exp.get_value(name_attachments, (id_exp, 1))

                    f0 = int(norm_f0*(last_frame-frame_attach)+frame_attach)
                    f1 = int(norm_f1*(last_frame-frame_attach)+frame_attach)

                    frames = list(x.loc[id_exp, f0:f1].index.get_level_values(id_frame_name))
                    exp_idx = list(np.full(len(frames), id_exp))
                    indexes += list(zip(exp_idx, frames))

                x2 = x.reindex(indexes)
                y2 = y.reindex(indexes)
                self.__compute_hist2d(x2, y2, x_intervals, y_intervals, result_names[i], labels[i], descriptions[i])

            self.exp.remove_object(name_x)
            self.exp.remove_object(name_y)
            self.exp.write(result_names)
        else:
            self.exp.load(result_names)

        plotter = Plotter(self.exp.root, self.exp.get_data_object(result_names[0]))
        nr = int(round(np.sqrt(len(dt_attach))))
        nc = int(round(len(dt_attach)/nr))
        fig, ax = plotter.create_plot(figsize=(10, 10), nrows=nr, ncols=nc, left=0.05)
        for i in range(len(dt_attach)-1):
            name = result_names[i]
            t0 = dt_attach[i]
            t1 = dt_attach[i+1]
            title = '['+str(t0)+', '+str(t1)+']'
            r = int(i/nc)
            c = i % nc

            plotter = Plotter(self.exp.root, self.exp.get_data_object(name))
            plotter.plot_heatmap(
                preplot=(fig, ax[r, c]), xlabel='', ylabel='', cbar_label='',
                title=title, normed=True, display_cbar=False)
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
        ax[1, 0].set_ylabel('Veracity')
        ax[-1, 1].set_xlabel('Confidence')

        plotter.save(fig, name=result_name)

    def w10s_smooth_food_direction_error_vs_path_efficiency_scatter_around_first_outside_attachment(self, redo=False):
        time = 10
        self.__plot_scatter_info_traj_around_first_outside_attachment(time, redo)

    def __plot_scatter_info_traj_around_first_outside_attachment(self, time, redo):
        result_name = 'w' + str(time) + \
                      's_smooth_food_direction_error_vs_path_efficiency_scatter_around_first_outside_attachment'
        label = 'Scatter plot of food path efficiency vs food direction error per experiment'
        description = 'Scatter plot of the 2 variables food path efficiency and food direction error per experiment ' \
                      'at different times'

        t_attach = range(-60, 120, 15)

        name_x = 'w' + str(time) + 's_food_path_efficiency'
        name_y = 'mm' + str(time) + 's_food_direction_error'
        name_attachments = 'outside_ant_attachment_frames'
        name_last_frame = 'food_exit_frames'

        if redo:

            self.exp.load([name_x, name_y, name_attachments, name_last_frame, 'fps'])
            self.exp.operation(name_y, lambda tab: 1 - (np.abs(tab) / np.pi))

            index_values = [(id_exp, t) for id_exp in self.exp.id_exp_list for t in t_attach]
            self.exp.add_new_empty_dataset(name=result_name, index_names=['id_exp', 'time'],
                                           column_names=[name_x, name_y], index_values=index_values,
                                           category=self.category, label=label, description=description)

            for id_exp in self.exp.id_exp_list:
                print(id_exp)

                x = self.exp.get_df(name_x).loc[id_exp, :]
                y = self.exp.get_df(name_y).loc[id_exp, :]
                frame_attach = self.exp.get_value(name_attachments, (id_exp, 1))
                last_frame = self.exp.get_value(name_last_frame, id_exp)
                fps = self.exp.get_value('fps', id_exp)

                x.index -= frame_attach
                y.index -= frame_attach
                x.index /= fps
                y.index /= fps

                for i in range(len(t_attach)-1):
                    t0 = t_attach[i]
                    t1 = t_attach[i+1]
                    x2 = np.around(np.mean(x.loc[t0:t1]), 5)
                    y2 = np.around(np.mean(y.loc[t0:t1]), 5)
                    self.exp.change_value(result_name, (id_exp, t0), [x2, y2])

                t0 = t_attach[-1]
                x2 = np.around(np.mean(x.loc[t0:last_frame]), 5)
                y2 = np.around(np.mean(y.loc[t0:last_frame]), 5)
                self.exp.change_value(result_name, (id_exp, t0), [x2, y2])

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(self.exp.root, self.exp.get_data_object(result_name))
        nr = int(round(np.sqrt(len(t_attach))))
        nc = int(round(len(t_attach)/nr))
        fig, ax = plotter.create_plot(figsize=(10, 10), nrows=nr, ncols=nc, left=0.05)
        for i in range(len(t_attach) - 1):
            t0 = t_attach[i]
            t1 = t_attach[i+1]
            title = '[' + str(t0) + ', ' + str(t1) + '] s'
            r = int(i/nc)
            c = i % nc

            df = self.exp.get_df(result_name).loc[pd.IndexSlice[:, t0], :]
            df.reset_index(inplace=True)
            df = df.drop(columns=['time', 'id_exp'])
            df.set_index(name_x, inplace=True)
            self.exp.add_new_dataset_from_df(df=df, name='to_plot', category=self.category, replace=True)

            plotter = Plotter(self.exp.root, self.exp.get_data_object('to_plot'))
            plotter.plot(preplot=(fig, ax[r, c]), xlabel='', ylabel='', ls='', title=title)
            ax[r, c].set_xlim((0, 1))
            ax[r, c].set_ylim((0, 1))

        t0 = t_attach[-1]
        title = '[' + str(t0) + ', inf[ s'
        df = self.exp.get_df(result_name).loc[pd.IndexSlice[:, t0], :]
        df.reset_index(inplace=True)
        df = df.drop(columns=['time', 'id_exp'])
        df.set_index(name_x, inplace=True)
        self.exp.add_new_dataset_from_df(df=df, name='to_plot', category=self.category, replace=True)
        plotter = Plotter(self.exp.root, self.exp.get_data_object('to_plot'))
        plotter.plot(preplot=(fig, ax[-1, -1]), xlabel='', ylabel='', ls='', title=title)
        ax[-1, -1].set_xlim((0, 1))
        ax[-1, -1].set_ylim((0, 1))

        ax[1, 0].set_ylabel('Veracity')
        ax[-1, 1].set_xlabel('Confidence')
        plotter.save(fig, name=result_name)

    def w10s_food_direction_error_vs_path_efficiency_velocity(self):
        time = 10
        name_x = 'w'+str(time)+'s_food_path_efficiency'
        name_y = 'mm'+str(time)+'s_food_direction_error'

        result_velocity_x_name = 'w'+str(time)+'s_food_direction_error_vs_path_efficiency_velocity_x'
        result_velocity_y_name = 'w'+str(time)+'s_food_direction_error_vs_path_efficiency_velocity_y'

        self.__compute_velocity_for_field_vector(name_x, name_y, result_velocity_x_name, result_velocity_y_name)

    def w30s_food_direction_error_vs_path_efficiency_velocity(self):
        time = 30
        name_x = 'w'+str(time)+'s_food_path_efficiency'
        name_y = 'mm'+str(time)+'s_food_direction_error'

        result_velocity_x_name = 'w'+str(time)+'s_food_direction_error_vs_path_efficiency_velocity_x'
        result_velocity_y_name = 'w'+str(time)+'s_food_direction_error_vs_path_efficiency_velocity_y'

        self.__compute_velocity_for_field_vector(name_x, name_y, result_velocity_x_name, result_velocity_y_name)

    def __compute_velocity_for_field_vector(self, name_x, name_y, result_velocity_x_name, result_velocity_y_name):
        label_x = 'X of food direction error vs path efficiency velocity'
        label_y = 'Y of food direction error vs path efficiency velocity'
        description_x = 'X coordinates of the velocity of the trajectory' \
                        ' taking food direction error as X and path efficiency velocity as Y'
        description_y = 'Y coordinates of the velocity of the trajectory' \
                        ' taking food direction error as X and path efficiency velocity as Y'

        self.exp.load([name_x, name_y, 'fps'])

        self.exp.add_copy1d(name_to_copy=name_x, copy_name=result_velocity_x_name, category=self.category,
                            label=label_x, description=description_x)
        self.exp.add_copy1d(name_to_copy=name_y, copy_name=result_velocity_y_name, category=self.category,
                            label=label_y, description=description_y)

        for id_exp in self.exp.characteristic_timeseries_exp_frame_index:
            fps = self.exp.get_value('fps', id_exp)

            dx = np.array(self.exp.get_df(name_x).loc[id_exp, :]).ravel()
            dx1 = dx[1].copy()
            dx2 = dx[-2].copy()
            dx[1:-1] = (dx[2:] - dx[:-2]) / 2.
            dx[0] = dx1 - dx[0]
            dx[-1] = dx[-1] - dx2

            dy = np.array(self.exp.get_df(name_y).loc[id_exp, :]).ravel()
            dy1 = dy[1].copy()
            dy2 = dy[-2].copy()
            dy[1:-1] = (dy[2:] - dy[:-2]) / 2.
            dy[0] = dy1 - dy[0]
            dy[-1] = dy[-1] - dy2

            self.exp.get_df(result_velocity_x_name).loc[id_exp, :] = np.around(dx * fps, 3)
            self.exp.get_df(result_velocity_y_name).loc[id_exp, :] = np.around(dy * fps, 3)

        self.exp.write(result_velocity_x_name)
        self.exp.write(result_velocity_y_name)

    def w10s_food_direction_error_vs_path_efficiency_vector_field(self, redo=False):
        time = 10
        dc = 0.1
        dv = 0.1
        self.__compute_field(time, dc, dv, redo)

    def w30s_food_direction_error_vs_path_efficiency_vector_field(self, redo=False):
        time = 30
        dc = 0.1
        dv = 0.1
        self.__compute_field(time, dc, dv, redo)

    def __compute_field(self, time, dc, dv, redo):
        result_name = 'w' + str(time) + 's_food_direction_error_vs_path_efficiency'
        result_name_x = result_name + '_vector_field_x'
        result_name_y = result_name + '_vector_field_y'
        result_name_norm = result_name + '_norm'

        confidence_intervals = np.arange(0, 1 + dc, dc)
        veracity_intervals = np.arange(0, 1 + dv, dv)

        if redo:

            confidence_intervals2 = np.around((confidence_intervals[1:] + confidence_intervals[:-1]) / 2., 2)
            veracity_intervals2 = np.around((veracity_intervals[1:] + veracity_intervals[:-1]) / 2., 2)

            name_confidence = 'w' + str(time) + 's_food_path_efficiency'
            name_veracity = 'mm' + str(time) + 's_food_direction_error'
            name_vect_x = 'w' + str(time) + 's_food_direction_error_vs_path_efficiency_velocity_x'
            name_vect_y = 'w' + str(time) + 's_food_direction_error_vs_path_efficiency_velocity_y'
            self.exp.load([name_vect_x, name_vect_y, name_confidence, name_veracity])

            label_x = 'X of the vector field food path efficiency vs food direction error'
            label_y = 'Y of the vector field food path efficiency vs food direction error'
            label_hist2d = '2D histogram of food path efficiency vs food direction error'

            description_x = 'X coordinates of the vector field of the process taking the food path efficiency as X and'\
                            ' food direction error as Y'
            description_y = 'Y coordinates of the vector field of the process taking the food path efficiency as X and'\
                            ' food direction error as Y'
            description_hist2d = 'Two-D histogram of the 2 variables food path efficiency and food direction error'

            self.exp.add_new_empty_dataset(name=result_name_x, index_names='confidence', fill_value=0,
                                           column_names=veracity_intervals2, index_values=confidence_intervals2,
                                           category=self.category, label=label_x, description=description_x)

            self.exp.add_new_empty_dataset(name=result_name_y, index_names='confidence', fill_value=0,
                                           column_names=veracity_intervals2, index_values=confidence_intervals2,
                                           category=self.category, label=label_y, description=description_y)

            self.exp.add_new_empty_dataset(name=result_name_norm, index_names='confidence', fill_value=0,
                                           column_names=veracity_intervals2, index_values=confidence_intervals2,
                                           category=self.category, label=label_hist2d, description=description_hist2d)

            tab_x, tab_y = self.__get_confidence_and_veracity_tab(
                name_confidence, name_veracity, confidence_intervals2, veracity_intervals2, dc, dv)

            tab_vect_x = self.exp.get_array(name_vect_x)
            tab_vect_y = self.exp.get_array(name_vect_y)

            for x in confidence_intervals2:
                mask = np.where(tab_x == x)[0]
                tab_y2 = tab_y[mask]
                tab_vect_x2 = tab_vect_x[mask]
                tab_vect_y2 = tab_vect_y[mask]
                for y in veracity_intervals2:
                    mask = np.where(tab_y2 == y)[0]
                    tab_vect_x3 = tab_vect_x2[mask]
                    tab_vect_y3 = tab_vect_y2[mask]

                    self.exp.get_df(result_name_x).loc[x, y] = np.nansum(tab_vect_x3)
                    self.exp.get_df(result_name_y).loc[x, y] = np.nansum(tab_vect_y3)
                    self.exp.get_df(result_name_norm).loc[x, y] = np.nansum(tab_vect_y3*0+1)

            self.exp.get_data_object(result_name_x).df /= self.exp.get_df(result_name_norm)
            self.exp.get_data_object(result_name_y).df /= self.exp.get_df(result_name_norm)

            self.exp.write([result_name_x, result_name_y, result_name_norm])

        else:
            self.exp.load([result_name_x, result_name_y, result_name_norm])
        tab_x = np.around(self.exp.get_index(result_name_x), 2)
        tab_y = np.array(self.exp.get_columns(result_name_x), dtype='float')

        plotter = Plotter(self.exp.root, self.exp.get_data_object(result_name_x))
        fig, ax = plotter.create_plot()

        self.exp.get_df(result_name_norm).index -= dc/2
        self.exp.get_df(result_name_norm).columns = np.array(self.exp.get_columns(result_name_norm), dtype=float) - dv/2
        plotter = Plotter(self.exp.root, self.exp.get_data_object(result_name_norm))
        plotter.plot_heatmap(preplot=(fig, ax), cbar_label='N. Data')

        mat_x, mat_y = np.meshgrid(tab_x, tab_y)
        mat_u = self.exp.get_data_object(result_name_x).get_array().T
        mat_v = self.exp.get_data_object(result_name_y).get_array().T
        # mat_norm = np.sqrt(mat_u ** 2 + mat_v ** 2)
        # mat_norm = np.sqrt(mat_norm)/mat_norm
        # mat_norm = np.minimum(mat_norm, 0.1)/mat_norm
        # mat_norm = (mat_norm)**(1/3.)/mat_norm
        mat_norm = 1
        ax.quiver(mat_x, mat_y, mat_u*mat_norm, mat_v*mat_norm, color='grey')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Veracity')
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_xticks(confidence_intervals)
        ax.set_yticks(veracity_intervals)
        plotter.save(fig, name=result_name)

    def w10s_food_direction_error_vs_path_efficiency_probability_matrix(self, redo=False):
        time = 10
        d = 0.1
        self.__compute_probability_matrix(time, d, d, redo)

    def w30s_food_direction_error_vs_path_efficiency_probability_matrix(self, redo=False):
        time = 30
        d = 0.1
        self.__compute_probability_matrix(time, d, d, redo)

    def __compute_probability_matrix(self, time, dc, dv, redo):
        result_name = 'w' + str(time) + 's_food_direction_error_vs_path_efficiency_probability_matrix'

        confidence_intervals = np.arange(0, 1 + dc, dc)
        veracity_intervals = np.arange(0, 1 + dv, dv)

        if redo:

            confidence_intervals2 = np.around((confidence_intervals[1:] + confidence_intervals[:-1]) / 2., 2)
            veracity_intervals2 = np.around((veracity_intervals[1:] + veracity_intervals[:-1]) / 2., 2)

            name_confidence = 'w' + str(time) + 's_food_path_efficiency'
            name_veracity = 'mm' + str(time) + 's_food_direction_error'
            self.exp.load([name_confidence, name_veracity, 'fps'])

            label = 'Probability transition of the markov process food direction error vs food path_efficiency '
            description = 'The trajectory (food direction error, food path_efficiency) ' \
                          'is considered as a Markov process. For each couple (confidence, veracity), is given ' \
                          'the probability to go to the North, i.e. to (confidence, veracity+dv), ' \
                          'the probability to go to the South, i.e. to (confidence, veracity-dv), ' \
                          'the probability to go to the East, i.e. to (confidence-dc, veracity), ' \
                          'the probability to go to the West, i.e. to (confidence+dc, veracity) and ' \
                          'the mean real time spend in this state (in seconds).'

            index_values = [(c, v) for c in confidence_intervals2 for v in veracity_intervals2]

            self.exp.add_new_empty_dataset(name=result_name, index_names=['confidence', 'veracity'], fill_value=0,
                                           column_names=['N', 'S', 'W', 'E', 'T'], index_values=index_values,
                                           category=self.category, label=label, description=description)

            tab_confidence, tab_veracity = self.__get_confidence_and_veracity_tab(
                name_confidence, name_veracity, confidence_intervals, veracity_intervals, dc, dv)

            for confidence in confidence_intervals2:
                for veracity in veracity_intervals2:
                    mask = np.where((tab_confidence == confidence) & (tab_veracity == veracity))[0]
                    t = len(mask)
                    self.exp.get_df(result_name).loc[(confidence, veracity)]['T'] = t

            tab_dconfidence = np.around(tab_confidence[1:]-tab_confidence[:-1], 2)
            tab_dveracity = np.around(tab_veracity[1:]-tab_veracity[:-1], 2)

            to_keep = ~np.isnan(tab_confidence[:-1]*tab_veracity[:-1]*tab_dconfidence*tab_dveracity) \
                & ((tab_dconfidence != 0) | (tab_dveracity != 0))

            mask = np.where(to_keep)[0]
            tab_confidence = tab_confidence[mask]
            tab_veracity = tab_veracity[mask]
            tab_dconfidence = tab_dconfidence[mask]
            tab_dveracity = tab_dveracity[mask]

            for confidence, veracity, dconfidence, dveracity \
                    in zip(tab_confidence, tab_veracity, tab_dconfidence, tab_dveracity):

                confidence = max(min(confidence,confidence_intervals2[-1]), confidence_intervals2[0])
                veracity = max(min(veracity, veracity_intervals2[-1]), veracity_intervals2[0])

                if dconfidence == 0 and dveracity > 0:
                    self.exp.get_df(result_name).loc[(confidence, veracity)]['N'] += 1
                elif dconfidence == 0 and dveracity < 0:
                    self.exp.get_df(result_name).loc[(confidence, veracity)]['S'] += 1
                elif dconfidence < 0 and dveracity == 0:
                    self.exp.get_df(result_name).loc[(confidence, veracity)]['W'] += 1
                elif dconfidence > 0 and dveracity == 0:
                    self.exp.get_df(result_name).loc[(confidence, veracity)]['E'] += 1

            self.exp.get_df(result_name)['T'] /= 100.

            self.exp.write(result_name)

        else:
            self.exp.load(result_name)

        df_norm = self.exp.get_df(result_name).copy()
        df_norm.drop(columns='T', inplace=True)
        df_norm = df_norm.sum(axis=1)

        for col in self.exp.get_columns(result_name):
            self.exp.get_data_object(result_name).df[col] /= df_norm

        tab_index = np.around(np.asarray(list(self.exp.get_index(result_name))), 2)
        tab_confidence = list(set(tab_index[:, 0]))
        tab_veracity = list(set(tab_index[:, 1]))
        tab_confidence.sort()
        tab_veracity.sort()
        mat_x, mat_y = np.meshgrid(tab_confidence, tab_veracity)

        mat_north = np.full((len(tab_confidence), len(tab_veracity)), np.nan)
        mat_south = np.full((len(tab_confidence), len(tab_veracity)), np.nan)
        mat_west = np.full((len(tab_confidence), len(tab_veracity)), np.nan)
        mat_east = np.full((len(tab_confidence), len(tab_veracity)), np.nan)
        self.exp.add_new_empty_dataset(name='time', index_names='confidence', column_names=np.array(tab_veracity)-dc/2.,
                                       index_values=np.array(tab_confidence)-dv/2., replace=True)
        self.exp.add_new_empty_dataset(name='norm', index_names='confidence', column_names=np.array(tab_veracity)-dc/2.,
                                       index_values=np.array(tab_confidence)-dv/2., replace=True)
        self.exp.add_new_empty_dataset(name='dist_uniform', index_names='confidence', column_names=np.array(tab_veracity)-dc/2.,
                                       index_values=np.array(tab_confidence)-dv/2., replace=True)

        for i, confidence in enumerate(tab_confidence):
            for j, veracity in enumerate(tab_veracity):
                nbr_data = df_norm.loc[(confidence, veracity)]

                if nbr_data > 5:
                    mat_north[i, j] = self.exp.get_df(result_name).loc[(confidence, veracity)]['N']
                    mat_south[i, j] = self.exp.get_df(result_name).loc[(confidence, veracity)]['S']
                    mat_west[i, j] = self.exp.get_df(result_name).loc[(confidence, veracity)]['W']
                    mat_east[i, j] = self.exp.get_df(result_name).loc[(confidence, veracity)]['E']

                    self.exp.get_df('time').loc[confidence-dc/2.][veracity-dv/2.] \
                        = self.exp.get_df(result_name).loc[(confidence, veracity)]['T']

                    self.exp.get_df('norm').loc[confidence-dc/2.][veracity-dv/2.] = nbr_data

                    dist_uniform = -(np.log2(mat_north[i, j]) + np.log2(mat_south[i, j])\
                        + np.log2(mat_west[i, j]) + np.log2(mat_east[i, j]))
                    self.exp.get_df('dist_uniform').loc[confidence-dc/2.][veracity-dv/2.] = dist_uniform

        plotter = Plotter(self.exp.root, self.exp.get_data_object('time'))
        fig, ax = plotter.create_plot()
        plotter.plot_heatmap(preplot=(fig, ax), cbar_label='Mean time (s)', vmin=0)
        headlength, headwidth, mat_zero, plotter, scale, width = self.__plot_proba_matrix(
            ax, confidence_intervals, veracity_intervals,
            mat_north, mat_south, mat_west, mat_east, mat_x, mat_y, result_name)
        plotter.save(fig, name=result_name)

        plotter = Plotter(self.exp.root, self.exp.get_data_object('norm'))
        fig, ax = plotter.create_plot()
        plotter.plot_heatmap(preplot=(fig, ax), cbar_label='N. data')
        headlength, headwidth, mat_zero, plotter, scale, width = self.__plot_proba_matrix(
            ax, confidence_intervals, veracity_intervals,
            mat_north, mat_south, mat_west, mat_east, mat_x, mat_y, result_name)
        plotter.save(fig, name=result_name, suffix='norm')

        plotter = Plotter(self.exp.root, self.exp.get_data_object('dist_uniform'))
        fig, ax = plotter.create_plot()
        plotter.plot_heatmap(preplot=(fig, ax), cbar_label='dist to uniform')
        headlength, headwidth, mat_zero, plotter, scale, width = self.__plot_proba_matrix(
            ax, confidence_intervals, veracity_intervals,
            mat_north, mat_south, mat_west, mat_east, mat_x, mat_y, result_name)
        plotter.save(fig, name=result_name, suffix='dist_uniform')

    def __plot_proba_matrix(self, ax, confidence_intervals, veracity_intervals, mat_north, mat_south, mat_west,
                            mat_east, mat_x, mat_y, result_name):
        plotter = Plotter(self.exp.root, self.exp.get_data_object(result_name))
        mat_zero = np.zeros(mat_north.shape)
        scale = 10
        headwidth = 1
        headlength = 1
        width = 0.01
        ax.quiver(mat_x, mat_y, mat_zero.T, mat_north.T, color='indigo', scale=scale, headwidth=headwidth,
                  width=width, headlength=headlength)
        ax.quiver(mat_x, mat_y, mat_zero.T, -mat_south.T, color='plum', scale=scale, headwidth=headwidth,
                  width=width, headlength=headlength)
        ax.quiver(mat_x, mat_y, -mat_west.T, mat_zero.T, color='mediumpurple', scale=scale, headwidth=headwidth,
                  width=width, headlength=headlength)
        ax.quiver(mat_x, mat_y, mat_east.T, mat_zero.T, color='thistle', scale=scale, headwidth=headwidth,
                  width=width, headlength=headlength)
        ax.quiver(mat_x, mat_y, mat_zero.T, mat_zero.T, color='k', scale=scale, minlength=2)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_xticks(confidence_intervals)
        ax.set_yticks(veracity_intervals)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Veracity')
        return headlength, headwidth, mat_zero, plotter, scale, width

    def __get_confidence_and_veracity_tab(
            self, name_confidence, name_veracity, confidence_intervals, veracity_intervals, dc, dv):

        mc = confidence_intervals[-1]
        self.exp.operation(name_confidence, lambda c: np.floor(c / dc) * dc + dc / 2.)
        self.exp.get_df(name_confidence)[self.exp.get_df(name_confidence) > mc] = mc
        tab_x = np.round(self.exp.get_array(name_confidence), 2)

        min_v = veracity_intervals[0]
        max_v = veracity_intervals[-1]
        self.exp.operation(name_veracity, lambda v: 1 - np.abs(v) / np.pi)
        self.exp.operation(name_veracity, lambda v: np.floor(v / dv) * dv + dv / 2.)
        self.exp.get_df(name_veracity)[self.exp.get_df(name_veracity) > max_v] = max_v
        self.exp.get_df(name_veracity)[self.exp.get_df(name_veracity) < min_v] = min_v
        tab_y = np.round(self.exp.get_array(name_veracity), 2)
        return tab_x, tab_y
