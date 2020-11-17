import numpy as np
import pandas as pd

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.MiscellaneousTools.ArrayManipulation import get_interval_containing, get_index_interval_containing
from Tools.Plotter.Plotter import Plotter
from Tools.Plotter.ColorObject import ColorObject


class AnalyseAttachments(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'Attachments'

    def compute_attachments(self):
        carrying_name = 'carrying'
        result_name = 'attachments'

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

    def compute_outside_attachments(self):
        carrying_name = 'carrying'
        from_outside_name = 'from_outside'
        result_name = 'outside_attachments'

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

    def compute_inside_attachments(self):
        carrying_name = 'carrying'
        from_outside_name = 'from_outside'
        result_name = 'inside_attachments'

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

        result_name = 'attachment_intervals'
        name = 'attachments'

        label = 'Between attachment intervals'
        description = 'Time intervals between attachment intervals (s)'

        self.__compute_attachment_intervals(description, label, name, redo, redo_hist, result_name)

    def compute_outside_attachment_intervals(self, redo=False, redo_hist=False):

        result_name = 'outside_attachment_intervals'
        name = 'outside_attachments'

        label = 'Between outside attachment intervals'
        description = 'Time intervals between outside attachment intervals (s)'

        self.__compute_attachment_intervals(description, label, name, redo, redo_hist, result_name)

    def compute_inside_attachment_intervals(self, redo=False, redo_hist=False):

        result_name = 'inside_attachment_intervals'
        name = 'inside_attachments'

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

    def compute_nbr_attachment_per_exp(self):
        outside_result_name = 'nbr_outside_attachment_per_exp'
        inside_result_name = 'nbr_inside_attachment_per_exp'
        result_name = 'nbr_attachment_per_exp'

        outside_attachment_name = 'outside_ant_carrying_intervals'
        non_outside_attachment_name = 'inside_ant_carrying_intervals'
        attachment_name = 'carrying_intervals'

        self.exp.load([attachment_name, non_outside_attachment_name, outside_attachment_name])

        self.exp.add_new1d_empty(name=outside_result_name, object_type='Characteristics1d', category=self.category,
                                 label='Number of outside attachments',
                                 description='Number of time an ant coming from outside attached to the food')

        self.exp.add_new1d_empty(name=inside_result_name, object_type='Characteristics1d', category=self.category,
                                 label='Number of inside attachments',
                                 description='Number of time an ant not coming from inside attached to the food')

        self.exp.add_new1d_empty(name=result_name, object_type='Characteristics1d', category=self.category,
                                 label='Number of attachments',
                                 description='Number of time an ant attached to the food')

        for id_exp in self.exp.id_exp_list:
            attachments = self.exp.get_df(attachment_name).loc[id_exp, :]
            self.exp.change_value(name=result_name, idx=id_exp, value=len(attachments))

            attachments = self.exp.get_df(outside_attachment_name).loc[id_exp, :]
            self.exp.change_value(name=outside_result_name, idx=id_exp, value=len(attachments))

            attachments = self.exp.get_df(non_outside_attachment_name).loc[id_exp, :]
            self.exp.change_value(name=inside_result_name, idx=id_exp, value=len(attachments))

        self.exp.write(result_name)
        self.exp.write(outside_result_name)
        self.exp.write(inside_result_name)

        bins = np.arange(0, 600, 25)

        for name in [result_name, outside_result_name, inside_result_name]:
            hist_name = self.exp.hist1d(name_to_hist=name, bins=bins)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
            fig, ax = plotter.plot()
            ax.grid()
            plotter.save(fig)

    def compute_first_attachment_time_of_outside_ant(self):
        result_name = 'first_attachment_time_of_outside_ant'
        carrying_name = 'outside_attachment_intervals'
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

    def compute_mean_food_direction_error_around_outside_ant_attachments(self, redo=False):
        result_name = 'mean_food_direction_error_around_first_outside_ant_attachments'
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
        result_name = 'mean_food_velocity_vector_length_around_first_outside_ant_attachments'

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

        res_vel_name = 'mean_food_velocity_vector_length_vs_direction_error_around_first_outside_attachments_X'
        res_error_name = 'mean_food_velocity_vector_length_vs_direction_error_around_first_outside_attachments_Y'

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

    def compute_food_traj_length_around_first_outside_attachment(self):
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

        bins = range(0, 130, 10)
        hist_name = self.exp.hist1d(name_to_hist=before_result_name, bins=bins)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot()
        ax.grid()
        ax.set_xticks(range(0, 120, 15))
        plotter.save(fig)

        bins = range(0, 500, 10)
        hist_name = self.exp.hist1d(name_to_hist=after_result_name, bins=bins)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot()
        ax.grid()
        ax.set_xticks(range(0, 430, 60))
        plotter.save(fig)

        surv_name = self.exp.survival_curve(name=after_result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(surv_name))
        fig, ax = plotter.plot()
        ax.grid()
        # ax.set_xticks(range(0, 430, 60))
        plotter.save(fig)

    def compute_inside_attachment_frames(self):
        result_name = 'inside_attachment_frames'
        carrying_name = 'inside_carrying_intervals'

        self.__get_attachment_frames(carrying_name, result_name)

    def compute_outside_attachment_frames(self):
        result_name = 'outside_attachment_frames'
        carrying_name = 'outside_carrying_intervals'

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

    def compute_nb_attachments_evol(self, redo=False):

        name = '%sattachment_intervals'
        result_name = 'nb_%sattachments_evol'
        init_frame_name = 'food_first_frame'

        dx = 0.05
        dx2 = 1.
        start_frame_intervals = np.arange(0, 2.5, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of %s attachments in a 1s period over time'
        description = 'Number of %s attaching to the food in a 1s period over time'

        cs = ['k', 'r', 'navy']

        for i, typ in enumerate(['', 'outside_', 'inside_']):

            self._get_nb_attachments_evol(name % typ, result_name % typ, init_frame_name,
                                          start_frame_intervals, end_frame_intervals,
                                          label % typ, description % typ, redo)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name % typ))
            fig, ax = plotter.plot_with_error(xlabel='Time (s)', ylabel='Attachment probability (for 1s)',
                                              label='probability', marker='', title='', c=cs[i])
            plotter.save(fig)

    def compute_nb_attachments_evol_around_first_outside_attachment(self, redo=False):

        name = '%sattachment_intervals'
        result_name = 'nb_%sattachments_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.05
        dx2 = 1/6.
        start_frame_intervals = np.arange(-1, 4, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of %s attachments in a 1s period over time'
        description = 'Number of %s attaching to the food in a 1s period over time'

        for typ in ['', 'outside_', 'inside_']:

            self._get_nb_attachments_evol(name % typ, result_name % typ, init_frame_name,
                                          start_frame_intervals, end_frame_intervals,
                                          label % typ, description % typ, redo)

        typ = ''
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name % typ))
        fig, ax = plotter.plot_with_error(
            xlabel='Time (s)', ylabel=r'$r$',
            label='Attachment rate', marker='', title='', c='k')
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name % typ), column_name='p')
        plotter.plot_fit(preplot=(fig, ax), typ='exp', cst=(-1, -1, 1), window=[2, 100])
        plotter.draw_vertical_line(ax)
        ax.set_xlim(-30, 120)
        ax.set_ylim(0, 1)
        plotter.save(fig)

        typ = 'outside_'
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name % typ))
        fig, ax = plotter.plot_with_error(
             xlabel='Time (s)', ylabel=r'$r_{out}$',
             label=r'$r_{out}$', marker='', title='', c='r')
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name % typ), column_name='p')
        plotter.plot_fit(preplot=(fig, ax), typ='exp', cst=(-.03, -.6, .8), window=[20, 60])
        plotter.plot()
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 1)
        plotter.save(fig)

        typ = 'inside_'
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name % typ))
        fig, ax = plotter.plot_with_error(
            xlabel='Time (s)', ylabel=r'$r_{in}$',
            label=r'$r_{in}$', marker='', title='', c='navy')
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name % typ), column_name='p')
        plotter.plot_fit(preplot=(fig, ax), typ='exp', cst=(-1, -1, 1), window=[10, 60])
        plotter.draw_vertical_line(ax)
        ax.set_xlim(-30, 120)
        ax.set_ylim(0, 1)
        plotter.save(fig)

    def _get_nb_attachments_evol(self, name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                 label, description, redo):
        if redo:

            self.exp.load([name, 'food_x', 'food_exit_frames'])

            self.cut_last_frames_for_indexed_by_exp_frame_indexed('food_x', 'food_exit_frames')
            self.cut_last_frames_for_indexed_by_exp_frame_indexed(name, 'food_exit_frames')

            self.change_first_frame(name, init_frame_name)
            self.change_first_frame('food_x', init_frame_name)

            # list_exp = [3, 12, 20, 30, 42, 45, 47, 49, 55]

            x = np.around(start_frame_intervals / 100., 2)
            y = np.full((len(start_frame_intervals), 3), np.nan)

            for i in range(len(start_frame_intervals)):
                frame0 = int(start_frame_intervals[i])
                frame1 = int(end_frame_intervals[i])

                # df = self.exp.get_df(name).loc[pd.IndexSlice[list_exp, :, frame0:frame1], :]
                # df_food = self.exp.get_df('food_x').loc[pd.IndexSlice[list_exp, frame0:frame1], :]
                df = self.exp.get_df(name).loc[pd.IndexSlice[:, :, frame0:frame1], :]
                df_food = self.exp.get_df('food_x').loc[pd.IndexSlice[:, frame0:frame1], :]

                n = round(len(df_food)/100)
                if n != 0:
                    p = np.around(len(df) / n, 3)
                    y[i, 0] = p
                    y[i, 1] = np.around(1.96*np.sqrt(p*(1-p)/n), 3)
                    y[i, 2] = np.around(1.96*np.sqrt(p*(1-p)/n), 3)

            mask = np.where(~np.isnan(y[:, 0]))[0]

            df = pd.DataFrame(y[mask, :], index=x[mask], columns=['p', 'err1', 'err2'])
            self.exp.add_new_dataset_from_df(df=df, name=result_name, category=self.category,
                                             label=label, description=description)
            self.exp.write(result_name)
            self.exp.remove_object(name)

        else:
            self.exp.load(result_name)

    def compute_ratio_inside_outside_attachments_evol_around_first_outside_attachment(self, redo=False):

        name_outside = 'outside_attachment_intervals'
        name_inside = 'inside_attachment_intervals'

        result_name = 'ratio_inside_outside_attachments_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.05
        dx2 = 1/6.
        start_frame_intervals = np.arange(0, 2.5, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Ratio between inside and outside attachments in a 1s period over time'
        description = 'Ratio between inside and outside attaching to the food in a 1s period over time'

        self._get_ratio_attachments_evol(name_outside, name_inside, result_name, init_frame_name,
                                         start_frame_intervals, end_frame_intervals,
                                         label, description, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel=r'$r_{in}/r_{out}$',
                               label='$r_{in}/r_{out}$', marker='', title='')
        plotter.plot_fit(preplot=(fig, ax), typ='exp', cst=(-1, 1, 1), window=[2, 50])

        ax.set_xlim((0, 120))
        ax.grid()
        plotter.save(fig)

    def _get_ratio_attachments_evol(self, name_outside, name_inside, result_name, init_frame_name,
                                    start_frame_intervals, end_frame_intervals,
                                    label, description, redo):
        if redo:

            self.exp.load([name_outside, name_inside, 'food_exit_frames'])

            self.cut_last_frames_for_indexed_by_exp_frame_indexed(name_outside, 'food_exit_frames')
            self.cut_last_frames_for_indexed_by_exp_frame_indexed(name_inside, 'food_exit_frames')

            self.change_first_frame(name_outside, init_frame_name)
            self.change_first_frame(name_inside, init_frame_name)

            # list_exp = [3, 12, 20, 30, 42, 45, 47, 49, 55]

            x = start_frame_intervals / 100.
            y = np.full(len(start_frame_intervals), np.nan)

            for i in range(len(start_frame_intervals)):
                frame0 = int(start_frame_intervals[i])
                frame1 = int(end_frame_intervals[i])

                # df = self.exp.get_df(name).loc[pd.IndexSlice[list_exp, :, frame0:frame1], :]
                # df_food = self.exp.get_df('food_x').loc[pd.IndexSlice[list_exp, frame0:frame1], :]
                df_outside = self.exp.get_df(name_outside).loc[pd.IndexSlice[:, :, frame0:frame1], :]
                df_inside = self.exp.get_df(name_inside).loc[pd.IndexSlice[:, :, frame0:frame1], :]

                if len(df_outside) + len(df_inside) != 0:
                    n_out = float(len(df_outside))
                    n_in = len(df_inside)
                    y[i] = n_in/n_out

            mask = np.where(~np.isnan(y))[0]

            index = pd.Index(np.around(x[mask], 2), name='time')
            df = pd.DataFrame(np.around(y[mask], 2), index=index, columns=['ratio'])
            self.exp.add_new_dataset_from_df(df=df, name=result_name, category=self.category,
                                             label=label, description=description)
            self.exp.write(result_name)
            self.exp.remove_object(name_outside)
            self.exp.remove_object(name_inside)

        else:
            self.exp.load(result_name)

    def _get_pulling_direction(self, result_name, name_attachment, label, description, bins, redo, redo_hist):
        if redo:
            name_exit = 'mm10_food_direction_error'
            init_frame_name = 'first_attachment_time_of_outside_ant'
            self.exp.load([name_attachment, name_exit, init_frame_name])

            self.exp.add_copy(old_name=name_attachment, new_name=result_name,
                              category=self.category, label=label, description=description)

            df_error = self.exp.get_df(name_exit).copy()
            df_error.reset_index(inplace=True)
            df_error[id_frame_name] -= 200
            df_error.set_index([id_exp_name, id_frame_name], inplace=True)
            index = self.exp.get_df(name_attachment).reset_index().set_index([id_exp_name, id_frame_name]).index
            df_error = df_error.reindex(index)
            self.exp.get_df(result_name)[:] = np.c_[df_error[:]]

            self.change_first_frame(result_name, init_frame_name)
            df_res = self.exp.get_df(result_name).loc[pd.IndexSlice[:, :, 0:], :]
            self.exp.change_df(result_name, df_res)

            self.exp.write(result_name)

        hist_name = self.compute_hist(result_name, bins=bins, redo=redo, redo_hist=redo_hist, error=True)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot_with_error(xlabel='(rad)', ylabel='PDF', title='', label='PDF')
        ax.set_ylim(0, 0.4)
        plotter.save(fig)

    def compute_pulling_direction_after_attachments(self, redo, redo_hist=False):
        dtheta = .4
        bins = np.arange(dtheta / 2., np.pi + dtheta / 2., dtheta)
        bins = np.array(list(-bins[::-1]) + list(bins))

        result_name = 'pulling_direction_after_%s_attachment'
        label = 'pulling direction after %s attachment'
        description = 'pulling direction after %s attachment'
        name_attachment = '%s_attachment_intervals'
        for suff in ['inside', 'outside']:
            self._get_pulling_direction(
                result_name % suff, name_attachment % suff, label % suff, description % suff, bins, redo, redo_hist)

        name = 'pulling_direction_after_%s_attachment_hist'

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(name % 'outside'))
        fig, ax = plotter.plot_with_error(
             xlabel='Pulling directions (rad)', ylabel='Probability', title='', label='outside', c='red')

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(name % 'inside'))
        plotter.plot_with_error(
            preplot=(fig, ax),
            xlabel='Pulling directions (rad)', ylabel='Probability', title='', label='inside', c='navy')

        plotter.draw_horizontal_line(ax, val=1/(2*np.pi)*dtheta, label='uniform')
        plotter.draw_legend(ax)
        ax.set_xlim(-np.pi, np.pi)
        # ax.set_ylim(0, .6)
        plotter.save(fig, name='pulling_direction_after_attachments')
