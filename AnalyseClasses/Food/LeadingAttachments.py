import numpy as np
import pandas as pd

from rdp import rdp

from DataStructure.VariableNames import id_exp_name, id_frame_name, id_ant_name
from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from Tools.Plotter.BasePlotters import BasePlotters
from Tools.Plotter.Plotter import Plotter
from Tools.MiscellaneousTools import Geometry as Geo, Fits


class AnalyseLeadingAttachments(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'LeadingAttachment'

    def get_leading_attachments(self):
        leading_name = 'is_leader'
        self.exp.load(leading_name)

        label = '%s leading attachment time series'
        description = 'Time series where 1 is when an %s ant attaches to the food ' \
                      'and change the food direction and 0 when not'
        result_name = '%sleading_attachments'
        food_name = '%sattachments'

        for suff in ['', 'outside_', 'inside_']:
            food_name2 = food_name % suff
            self.exp.load(food_name2)
            result_name2 = result_name % suff
            self.exp.add_copy(old_name=food_name2, new_name=result_name2, category=self.category,
                              label=label % suff, description=description % suff)
            self.exp.get_df(result_name2).columns = [result_name2]

            def do4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                frame = df.index.get_level_values(id_frame_name)[0]
                print(id_exp, frame)
                self.exp.get_df(result_name2).loc[id_exp, frame, :] *= int(df.iloc[0])

            self.exp.groupby(leading_name, [id_exp_name, id_frame_name], do4each_group)
            self.exp.change_df(result_name2, self.exp.get_df(result_name2).astype(int))

            self.exp.write(result_name2)

    def compute_leading_attachment_intervals(self, redo=False, redo_hist=False):

        result_name = '%sleading_attachment_intervals'
        name = '%sleading_attachments'

        label = 'Between %s leading attachment intervals'
        description = 'Time intervals between %s leading attachment intervals (s)'

        for suff in ['', 'outside_', 'inside_']:
            self.__compute_attachment_intervals(
                name % suff, result_name % suff, label % suff, description % suff, redo, redo_hist)

    def __compute_attachment_intervals(self, name, result_name, label, description, redo, redo_hist):
        bins = np.arange(0, 100, 3)
        if redo is True:
            self.exp.load(name)
            df = 1-self.exp.get_df(name)
            df.index = df.index.droplevel('id_ant')
            self.exp.add_new1d_from_df(df, 'temp', 'CharacteristicTimeSeries1d', replace=True)

            temp_name = self.exp.compute_time_intervals(name_to_intervals='temp', replace=True)

            df_attach = self.exp.get_df(name).copy()
            df_attach.loc[:, :, -1] = np.nan
            df_attach[df_attach == 0] = np.nan
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
        plotter.plot_fit(typ='exp', preplot=(fig, ax), window=[0, 15])
        plotter.save(fig)

    def compute_nb_leading_attachment_per_outside_ant(self, redo, redo_hist=False):
        result_name = 'nb_leading_attachment_per_outside_ant'
        label = 'Number of attachments'
        description = 'Number of leading attachments per outside ant'
        bins = np.arange(0.5, 11, 1)

        if redo is True:

            name = 'leading_attachment_intervals'
            from_outside_name = 'from_outside'

            self.exp.load([name, from_outside_name])

            res = []

            def do4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                id_ant = df.index.get_level_values(id_ant_name)[0]

                from_outside = self.exp.get_value(from_outside_name, (id_exp, id_ant))

                if from_outside == 1:
                    frames = df.index.get_level_values(id_frame_name)
                    res.append((id_exp, id_ant, len(frames)))

                return df

            self.exp.groupby(name, [id_exp_name, id_ant_name], do4each_group)

            self.exp.add_new_dataset_from_array(array=np.array(res), name=result_name,
                                                index_names=[id_exp_name, id_ant_name], column_names=result_name,
                                                category=self.category, label=label, description=description)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel=r'Number', ylabel='Occurrences', normed=False, display_legend=False, title='')
        ax.grid()
        plotter.save(fig)

    def compute_nb_leading_attachments_evol(self, redo=False):

        typ = ''

        name = 'leading_attachment_intervals'
        result_name = 'nb_leading_attachments_evol'
        init_frame_name = 'food_first_frame'

        dx = 0.05
        dx2 = 1.
        start_frame_intervals = np.arange(0, 2.5, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of leading %s attachments in a 1s period over time'
        description = 'Number of leading %s attaching to the food in a 1s period over time'

        self._get_nb_attachments_evol(name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                      label % typ, description % typ, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot_with_error(xlabel='Time (s)', ylabel='Attachment probability (for 1s)',
                                          label='probability', marker='', title='')
        plotter.save(fig)

    def compute_nb_leading_attachments_evol_around_first_outside_attachment(self, redo=False):

        typ = ''
        name = 'leading_attachment_intervals'

        result_name = 'nb_leading_attachments_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.05
        dx2 = 1.
        start_frame_intervals = np.arange(0, 2.5, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of leading %s attachments in a 1s period over time'
        description = 'Number of leading %s attaching to the food in a 1s period over time'

        self._get_nb_attachments_evol(name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                      label % typ, description % typ, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot_with_error(xlabel='Time (s)', ylabel='Attachment probability (for 1s)',
                                          label='probability', marker='', title='')
        plotter.save(fig)

    def compute_nb_leading_outside_attachments_evol(self, redo=False):

        typ = 'outside'

        name = 'outside_leading_attachment_intervals'
        result_name = 'nb_outside_leading_attachments_evol'
        init_frame_name = 'food_first_frame'

        dx = 0.05
        dx2 = 1.
        start_frame_intervals = np.arange(0, 2.5, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of leading %s attachments in a 1s period over time'
        description = 'Number of leading %s attaching to the food in a 1s period over time'

        self._get_nb_attachments_evol(name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                      label % typ, description % typ, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot_with_error(xlabel='Time (s)', ylabel='Attachment probability (for 1s)',
                                          label='probability', marker='', title='')
        plotter.save(fig)

    def compute_nb_leading_outside_attachments_evol_around_first_outside_attachment(self, redo=False):

        typ = 'outside'
        name = 'outside_leading_attachment_intervals'

        result_name = 'nb_outside_leading_attachments_evol_around_first_outside_attachment'
        init_frame_name = 'first_leading_attachment_time_of_outside_ant'

        dx = 0.05
        dx2 = 1/6.
        start_frame_intervals = np.arange(-1, 4, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of leading %s attachments in a 1s period over time'
        description = 'Number of leading %s attaching to the food in a 1s period over time'

        self._get_nb_attachments_evol(name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                      label % typ, description % typ, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot_with_error(xlabel='Time (s)', ylabel='Attachment probability (for 1s)',
                                          label='probability', marker='', title='')
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name='p')
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 0.15)
        plotter.draw_horizontal_line(ax, val=0.057, c='w', ls='-', label='y=0.057')
        plotter.draw_legend(ax)
        plotter.save(fig)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot_with_error(xlabel='Time (s)', ylabel=r'$p_{out}$',
                                          label='probability', marker='', title='')
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name='p')
        plotter.plot_fit(preplot=(fig, ax), typ='exp', cst=(-1, -1, 1), window=[2, 100])
        # lim = 20
        # plotter.plot_fit(preplot=(fig, ax), typ='linear', window=[2, lim], c='orange')
        # plotter.plot_fit(preplot=(fig, ax), typ='cst', window=[lim, 100], c='red')
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 0.14)
        plotter.draw_legend(ax)
        ax.grid()
        plotter.save(fig, suffix='2')

    def compute_nb_leading_inside_attachments_evol_around_first_outside_attachment(self, redo=False):

        typ = 'inside'
        name = 'inside_leading_attachment_intervals'

        result_name = 'nb_inside_leading_attachments_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.05
        dx2 = 1/6.
        start_frame_intervals = np.arange(-1, 2.5, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of leading %s attachments in a 1s period over time'
        description = 'Number of leading %s attaching to the food in a 1s period over time'

        self._get_nb_attachments_evol(name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                      label % typ, description % typ, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot_with_error(xlabel='Time (s)', ylabel=r'$p_{in}$',
                                          label='probability', marker='', title='')
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name='p')
        plotter.plot_fit(preplot=(fig, ax), typ='exp', cst=(-1, -1, 1), window=[20, 100])

        ax.set_xlim(0, 80)
        ax.set_ylim(0, .14)
        plotter.draw_vertical_line(ax)
        ax.grid()
        plotter.save(fig)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot_with_error(xlabel='Time (s)', ylabel=r'$p_{in}(t)$',
                                          label=r'$p_{in}$', marker='', title='')
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name='p')
        plotter.plot_fit(preplot=(fig, ax), typ='exp', cst=(-1, -1, 1), window=[20, 100])
        x = np.arange(0, 150, 0.1)
        p_in = Fits.exp_fct(x, -0.062, 3.1, 0.64)*0.057
        ax.plot(x, p_in, label=r'$(1-q)/q(t) p_{out}$', c='navy')

        ax.set_xlim(0, 80)
        ax.set_ylim(0, .25)
        plotter.draw_vertical_line(ax)
        ax.grid()
        plotter.draw_legend(ax)
        plotter.save(fig, suffix='2')

    def _get_nb_attachments_evol(self, name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                 label, description, redo):
        if redo:

            self.exp.load([name, 'food_x', 'food_exit_frames'])

            self.cut_last_frames_for_indexed_by_exp_frame_indexed('food_x', 'food_exit_frames')
            self.cut_last_frames_for_indexed_by_exp_frame_indexed(name, 'food_exit_frames')

            self.change_first_frame(name, init_frame_name)
            self.change_first_frame('food_x', init_frame_name)

            # list_exp = [3, 12, 20, 30, 42, 45, 47, 49, 55]

            x = np.around(start_frame_intervals / 2. / 100., 2)
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

    def compute_ratio_leading_inside_outside_attachments_evol_around_first_outside_attachment(self, redo=False):

        name_outside = 'outside_leading_attachment_intervals'
        name_inside = 'inside_leading_attachment_intervals'

        result_name = 'ratio_inside_outside_leading_attachments_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.05
        dx2 = 1/6.
        start_frame_intervals = np.arange(0, 2.5, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Ratio between inside and outside leading attachments in a 1s period over time'
        description = 'Ratio between inside and outside leading attaching to the food in a 1s period over time'

        self._get_ratio_attachments_evol(name_outside, name_inside, result_name, init_frame_name,
                                         start_frame_intervals, end_frame_intervals,
                                         label, description, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel=r'$p_{out}/(p_{out}+p_{in})$',
                               label='probability', marker='', title='')
        plotter.plot_fit(preplot=(fig, ax), typ='exp', cst=(-1, 1, 1), window=[0, 300])

        # lim = 100
        # plotter.plot_fit(preplot=(fig, ax), typ='linear', window=[10, lim+2], c='orange')
        # plotter.plot_fit(preplot=(fig, ax), typ='cst', window=[lim, 200], c='red')
        ax.set_xlim((0, 80))
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

            x = start_frame_intervals / 2. / 100.
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
                    y[i] = n_out/(n_out+n_in)

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

    def compute_prop_leading_attachments_evol(self, redo=False):

        for typ in ['outside', 'inside']:
            name_all = '%s_attachment_intervals'
            name_leading = '%s_leading_attachment_intervals'
            result_name = 'prop_%s_leading_attachments_evol'
            init_frame_name = 'first_attachment_time_of_outside_ant'

            dx = 0.05
            dx2 = 2 / 6.
            start_frame_intervals = np.arange(-1, 4, dx) * 60 * 100
            end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

            label = 'Proportion of leading %s attachments over time'
            description = 'Number of leading %s attachment over the total number of attachment to the food over time'

            self._get_prop_attachments_evol(name_leading % typ, name_all % typ, result_name % typ, init_frame_name,
                                            start_frame_intervals, end_frame_intervals,
                                            label % typ, description % typ, redo)

        typ = 'outside'
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name % typ))
        fig, ax = plotter.plot_with_error(xlabel='Time (s)', ylabel='%s leading attachment proportion' % typ,
                                          label='proportion %s' % typ, marker='', title='', c='r')
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name % typ), column_name='p')
        plotter.plot_fit(preplot=(fig, ax), typ='exp', cst=(-.01, .1, .1), window=[8, 65])
        ax.set_xlim(0, 80)
        ax.set_ylim(0, .4)
        plotter.draw_vertical_line(ax)
        plotter.save(fig)

        typ = 'inside'
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name % typ))
        fig, ax = plotter.plot_with_error(xlabel='Time (s)', ylabel='%s leading attachment proportion' % typ,
                                          label='proportion %s' % typ, marker='', title='', c='navy')
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name % typ), column_name='p')
        plotter.plot_fit(preplot=(fig, ax), typ='exp', cst=(-1, -1, 1), window=[20, 72])
        ax.set_xlim(-30, 80)
        ax.set_ylim(0, .4)
        plotter.draw_vertical_line(ax)
        plotter.save(fig)

    def _get_prop_attachments_evol(self, name_leading, name_all, result_name, init_frame_name,
                                   start_frame_intervals, end_frame_intervals, label, description, redo):
        if redo:

            self.exp.load([name_leading, name_all, 'food_exit_frames'])

            self.cut_last_frames_for_indexed_by_exp_frame_indexed(name_all, 'food_exit_frames')
            self.cut_last_frames_for_indexed_by_exp_frame_indexed(name_leading, 'food_exit_frames')

            self.change_first_frame(name_all, init_frame_name)
            self.change_first_frame(name_leading, init_frame_name)

            # list_exp = [3, 12, 20, 30, 42, 45, 47, 49, 55]

            x = np.around(start_frame_intervals / 2. / 100., 2)
            y = np.full((len(start_frame_intervals), 3), np.nan)

            for i in range(len(start_frame_intervals)):
                frame0 = int(start_frame_intervals[i])
                frame1 = int(end_frame_intervals[i])

                # df_all = self.exp.get_df(name_all).loc[pd.IndexSlice[list_exp, :, frame0:frame1], :]
                # df_leading = self.exp.get_df(name_leading).loc[pd.IndexSlice[list_exp, :, frame0:frame1], :]
                df_all = self.exp.get_df(name_all).loc[pd.IndexSlice[:, :, frame0:frame1], :]
                df_leading = self.exp.get_df(name_leading).loc[pd.IndexSlice[:, :, frame0:frame1], :]

                n = len(df_all)
                if n != 0:
                    p = np.around(len(df_leading) / n, 3)
                    y[i, 0] = p
                    y[i, 1] = np.around(1.96*np.sqrt(p*(1-p)/n), 3)
                    y[i, 2] = np.around(1.96*np.sqrt(p*(1-p)/n), 3)

            mask = np.where(~np.isnan(y[:, 0]))[0]

            df_all = pd.DataFrame(y[mask, :], index=x[mask], columns=['p', 'err1', 'err2'])
            self.exp.add_new_dataset_from_df(df=df_all, name=result_name, category=self.category,
                                             label=label, description=description)
            self.exp.write(result_name)
            self.exp.remove_object(name_all)
            self.exp.remove_object(name_leading)

        else:
            self.exp.load(result_name)

    def compute_nb_attachments_evol_w10s_path_efficiency(self, redo=False):

        discrim_name = 'w10s_food_path_efficiency'
        name = '%s_attachment_intervals'
        result_name = 'nb_%s_attachments_evol_%s'
        d_eff = 0.05
        start_eff_intervals = np.around(np.arange(0, 1, d_eff), 2)
        end_eff_intervals = np.around(start_eff_intervals+d_eff, 2)

        label = 'Number of %s attachments in a 1s period over time'
        description = 'Number of %s attaching to the food in a 1s period over time'

        for typ in ['outside', 'inside']:
            self._get_nb_attachments_evol_path_efficiency(
                name % typ, discrim_name, result_name % (typ, discrim_name), start_eff_intervals,
                end_eff_intervals, label % typ, description % typ, redo)

        suff = 'outside'
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name % (suff, discrim_name)))
        fig, ax = plotter.plot_with_error(xlabel='Path efficiency', ylabel='Attachment probability (for 1s)',
                                          label='probability', marker='', title='', c='r')
        plotter.save(fig)
        suff = 'inside'
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name % (suff, discrim_name)))
        fig, ax = plotter.plot_with_error(xlabel='Path efficiency', ylabel='Attachment probability (for 1s)',
                                          label='probability', marker='', title='', c='navy')
        plotter.save(fig)

    def compute_nb_leading_attachments_evol_w10s_path_efficiency(self, redo=False):

        discrim_name = 'w10s_food_path_efficiency'
        name = '%s_leading_attachment_intervals'
        result_name = 'nb_%s_leading_attachments_evol_%s'
        d_eff = 0.05
        start_eff_intervals = np.around(np.arange(0, 1, d_eff), 2)
        end_eff_intervals = np.around(start_eff_intervals+d_eff, 2)

        label = 'Number of leading %s attachments in a 1s period over time'
        description = 'Number of leading %s attaching to the food in a 1s period over time'

        for typ in ['outside', 'inside']:
            self._get_nb_attachments_evol_path_efficiency(
                name % typ, discrim_name, result_name % (typ, discrim_name), start_eff_intervals,
                end_eff_intervals, label % typ, description % typ, redo)

        suff = 'outside'
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name % (suff, discrim_name)))
        fig, ax = plotter.plot_with_error(xlabel='Path efficiency', ylabel='Attachment probability (for 1s)',
                                          label='probability', marker='', title='', c='r')
        plotter.save(fig)
        suff = 'inside'
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name % (suff, discrim_name)))
        fig, ax = plotter.plot_with_error(xlabel='Path efficiency', ylabel='Attachment probability (for 1s)',
                                          label='probability', marker='', title='', c='navy')
        plotter.save(fig)

    def _get_nb_attachments_evol_path_efficiency(self, name, discrim_name, result_name, start_eff_intervals,
                                                 end_eff_intervals, label, description, redo):
        if redo:

            self.exp.load([name, discrim_name])

            df_discrim = self.exp.get_df(discrim_name).copy()
            df_discrim.reset_index(inplace=True)
            df_discrim[id_frame_name] += 500
            df_discrim.set_index([id_exp_name, id_frame_name], inplace=True)
            df_discrim = df_discrim.dropna()
            index = df_discrim.index

            df_res = self.exp.get_df(name).copy()
            df_res.reset_index(inplace=True)
            df_res.pop(id_ant_name)
            df_res.set_index([id_exp_name, id_frame_name], inplace=True)
            df_res[:] = 1
            df_res = df_res.reindex(index, fill_value=0)

            df_res[discrim_name] = np.c_[np.around(df_discrim.values.ravel(), 3)]
            df_res.reset_index(inplace=True)
            df_res.pop(id_frame_name)
            df_res.set_index([id_exp_name, discrim_name], inplace=True)
            df_res.sort_index(inplace=True)

            x = np.around(start_eff_intervals, 2)
            y = np.full((len(start_eff_intervals), 3), np.nan)

            for i in range(len(start_eff_intervals)):
                eff0 = start_eff_intervals[i]
                eff1 = end_eff_intervals[i]

                df = df_res.loc[pd.IndexSlice[:, eff0:eff1], :]
                n = len(df) / 100.
                if n > 0:
                    p = np.around(np.mean(df), 10) * 100
                    y[i, 0] = p
                    y[i, 1] = np.around(1.96 * np.sqrt(p * (1 - p) / n), 10)
                    y[i, 2] = np.around(1.96 * np.sqrt(p * (1 - p) / n), 10)

            mask = np.where(~np.isnan(y[:, 0]))[0]

            df_res = pd.DataFrame(y[mask, :], index=x[mask], columns=['p', 'err1', 'err2'])
            self.exp.add_new_dataset_from_df(df=df_res, name=result_name, category=self.category,
                                             label=label, description=description)
            self.exp.write(result_name)
            self.exp.remove_object(name)
            self.exp.remove_object(discrim_name)

        else:
            self.exp.load(result_name)

    def compute_first_leading_attachment_time_of_outside_ant(self):
        result_name = 'first_leading_attachment_time_of_outside_ant'
        carrying_name = 'outside_leading_attachment_intervals'
        self.exp.load(carrying_name)

        self.exp.add_new1d_empty(name=result_name, object_type='Characteristics1d',
                                 category=self.category, label='First leading attachment time of an outside ant',
                                 description='First leading attachment time of an ant coming from outside')

        def compute_first_attachment4each_exp(df):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            frames = df.index.get_level_values(id_frame_name)
            print(id_exp)

            min_time = int(frames.min())
            self.exp.change_value(result_name, id_exp, min_time)

        self.exp.get_df(carrying_name).groupby(id_exp_name).apply(compute_first_attachment4each_exp)
        self.exp.write(result_name)

    def compute_first_leading_attachment_time(self):
        result_name = 'first_leading_attachment_time'
        carrying_name = 'leading_attachment_intervals'
        self.exp.load(carrying_name)

        self.exp.add_new1d_empty(name=result_name, object_type='Characteristics1d',
                                 category=self.category, label='First leading attachment time of an outside ant',
                                 description='First leading attachment time of an ant')

        def compute_first_attachment4each_exp(df):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            frames = df.index.get_level_values(id_frame_name)
            print(id_exp)

            min_time = int(frames.min())
            self.exp.change_value(result_name, id_exp, min_time)

        self.exp.get_df(carrying_name).groupby(id_exp_name).apply(compute_first_attachment4each_exp)
        self.exp.write(result_name)

    def compute_isolated_attachments(self):
        max_dt0 = 2
        max_dt1 = 2

        for suff in ['outside', 'inside']:
            self._get_isolated_attachments(suff, max_dt0, max_dt1)

    def _get_isolated_attachments(self, suff, max_dt0, max_dt1):
        name = 'leading_attachment_intervals'
        name_inoutside = '%s_leading_attachment_intervals' % suff
        self.exp.load([name, name_inoutside])
        result_name = 'isolated_%s_leading_attachment_intervals' % suff
        label = 'isolated %s attachment intervals' % suff
        description = 'Time intervals between %s leading attachments isolated in time from attachments,' \
                      ' i.e. no other leading attachment occurs ' \
                      'between %s before and %s after' % (suff, max_dt0, max_dt1)
        self.exp.add_copy(
            old_name=name_inoutside, new_name=result_name, category=self.category, label=label, description=description)

        def do4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]

            df2 = df.loc[id_exp, :, :].reset_index()
            df2 = df2.set_index([id_frame_name, id_ant_name]).sort_index()

            df3 = self.exp.get_df(name).loc[id_exp, :, :].reset_index()
            df3 = df3.set_index([id_exp_name, id_frame_name, id_ant_name]).loc[id_exp, :, :].sort_index()

            for frame, id_ant, _ in df2.reset_index().values:
                if (frame, id_ant) in df3.index:
                    dt1 = float(df3.loc[frame, id_ant])
                    if dt1 < max_dt1:
                        df.loc[id_exp, id_ant, frame] = np.nan
                    else:
                        df_temp = df3.loc[:frame, :]
                        if len(df_temp) > 1:
                            dt0 = float(df_temp.iloc[-2])
                            if dt0 < max_dt0:
                                df.loc[id_exp, id_ant, frame] = np.nan
            return df

        df_res = self.exp.groupby(name_inoutside, id_exp_name, do4each_group)
        df_res.dropna(inplace=True)
        print(name_inoutside, len(df_res))
        self.exp.change_df(result_name, df_res)
        self.exp.write(result_name)

    def compute_nb_isolated_leading_attachments_evol_around_first_outside_attachment(self, redo=False):

        name = 'isolated_%s_leading_attachment_intervals'

        result_name = 'nb_isolated_%s_leading_attachments_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'
        dx = 0.05
        dx2 = 1/6.
        start_frame_intervals = np.arange(-1, 2.5, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of leading %s attachments in a 1s period over time'
        description = 'Number of leading %s attaching to the food in a 1s period over time'

        for typ in ['outside', 'inside']:

            self._get_nb_attachments_evol(
                name % typ, result_name % typ, init_frame_name,
                start_frame_intervals, end_frame_intervals, label % typ, description % typ, redo)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name % typ))
            fig, ax = plotter.plot_with_error(xlabel='Time (s)', ylabel='Attachment probability (for 1s)',
                                              label='probability', marker='', title='')
            plotter.save(fig)

    def compute_pulling_direction_after_leading_attachments(self, redo, redo_hist=False):
        dtheta = .4
        bins = np.arange(dtheta / 2., np.pi + dtheta / 2., dtheta)
        bins = np.array(list(-bins[::-1]) + list(bins))

        result_name = 'pulling_direction_after_%s_leading_attachment'
        label = 'pulling direction after %s leading_attachment'
        description = 'pulling direction after %s leading_attachment'
        name_attachment = '%s_leading_attachment_intervals'
        for suff in ['inside', 'outside']:
            self._get_pulling_direction(
                result_name % suff, name_attachment % suff, label % suff, description % suff, bins, redo, redo_hist)

        name = 'pulling_direction_after_%s_leading_attachment_hist'

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
        plotter.save(fig, name='pulling_direction_after_leading_attachments')

    def compute_pulling_direction_after_isolated_leading_attachments(self, redo, redo_hist=False):
        dtheta = .4
        bins = np.arange(dtheta / 2., np.pi + dtheta / 2., dtheta)
        bins = np.array(list(-bins[::-1]) + list(bins))
        result_name = 'pulling_direction_after_isolated_%s_leading_attachment'
        label = 'pulling direction after isolated %s leading_attachment'
        description = 'pulling direction after isolated %s leading_attachment'
        name_attachment = 'isolated_%s_leading_attachment_intervals'
        for suff in ['inside', 'outside']:
            self._get_pulling_direction(
                result_name % suff, name_attachment % suff, label % suff, description % suff, bins, redo, redo_hist)

        name = 'pulling_direction_after_isolated_%s_leading_attachment_hist'

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
        plotter.save(fig, name='pulling_direction_after_isolated_leading_attachments')

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

    def compute_pulling_direction_after_isolated_leading_attachments2(self, redo, redo_hist=False):

        for suff in ['inside', 'outside']:

            result_name = 'pulling_direction_after_isolated_%s_leading_attachment2' % suff
            label = 'pulling direction after isolated %s leading_attachment' % suff
            description = 'pulling direction after isolated %s leading_attachment' % suff
            name_attachment = 'isolated_%s_leading_attachment_intervals' % suff
            self._get_pulling_direction2(result_name, name_attachment, label, description, redo, redo_hist)

    def compute_pulling_direction_after_leading_attachments2(self, redo, redo_hist=False):

        for suff in ['inside', 'outside']:

            result_name = 'pulling_direction_after_%s_leading_attachment2' % suff
            label = 'pulling direction after %s leading_attachment' % suff
            description = 'pulling direction after %s leading_attachment' % suff
            name_attachment = '%s_leading_attachment_intervals' % suff
            self._get_pulling_direction2(result_name, name_attachment, label, description, redo, redo_hist)

    def _get_pulling_direction2(self, result_name, name_attachment, label, description, redo, redo_hist):
        dtheta = .25
        bins = np.arange(0, np.pi + dtheta / 2., dtheta)
        # bins = np.array(list(-bins[::-1]) + list(bins))
        if redo:

            name_exit = 'mm1s_food_direction_error'
            init_frame_name = 'first_attachment_time_of_outside_ant'
            self.exp.load([name_attachment, name_exit, init_frame_name])

            index = self.exp.get_df(name_attachment).reset_index().set_index([id_exp_name, id_frame_name]).index
            index = index.sort_values()
            index_names = [id_exp_name, id_frame_name]
            self.exp.add_new_empty_dataset(name=result_name, index_names=index_names,
                                           column_names=['before', 'after'], index_values=index,
                                           category=self.category, label=label, description=description)

            df_error2 = self.exp.get_df(name_exit).copy()
            df_error2.reset_index(inplace=True)
            df_error2[id_frame_name] -= 200
            df_error2.set_index([id_exp_name, id_frame_name], inplace=True)

            df_error1 = self.exp.get_df(name_exit).copy()

            df_error2 = df_error2.reindex(index)
            df_error1 = df_error1.reindex(index)

            self.exp.get_df(result_name)['before'] = np.abs(np.around(np.c_[df_error1[:]], 3))
            self.exp.get_df(result_name)['after'] = np.abs(np.around(np.c_[df_error2[:]], 3))

            self.change_first_frame(result_name, init_frame_name)
            # df_res = self.exp.get_df(result_name).loc[pd.IndexSlice[:, 0:], :]
            df_res = self.exp.get_df(result_name)
            self.exp.change_df(result_name, df_res)

            self.exp.write(result_name)

        hist_name = self.compute_hist(result_name, bins=bins, redo=redo, redo_hist=redo_hist, error=True)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='(rad)', ylabel='PDF', title='', normed=True)
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, 1.)
        plotter.save(fig)

    def compute_pulling_direction_after_leading_attachments_evol(self, redo):
        for suff in ['outside', 'inside']:
            name = 'pulling_direction_after_%s_leading_attachment2' % suff
            result_name = name+'_hist_evol'

            dtheta = np.pi/10.
            bins = np.arange(0, np.pi+dtheta, dtheta)

            dx = 0.5
            start_frame_intervals = np.array(np.arange(-1, 2., dx)*60*100, dtype=int)
            end_frame_intervals = np.array(start_frame_intervals + dx*60*100, dtype=int)

            label = 'pulling direction after %s leading_attachment distribution over time (rad)' % suff
            description = 'Histogram over time of the pulling directions after %s leading_attachment' % suff

            if redo:
                self.exp.load(name)

                self.exp.operation(name, lambda a: np.abs(a))
                self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                          end_frame_intervals=end_frame_intervals, bins=bins,
                                          result_name=result_name+'_after', category=self.category,
                                          label=label, description=description, column_to_hist='after')
                self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                          end_frame_intervals=end_frame_intervals, bins=bins,
                                          result_name=result_name+'_before', category=self.category,
                                          label=label, description=description, column_to_hist='before')
                self.exp.remove_object(name)
                self.exp.write(result_name+'_after')
                self.exp.write(result_name+'_before')
            else:
                self.exp.load(result_name)

            for suff2 in ['before', 'after']:
                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name+'_'+suff2))
                fig, ax = plotter.plot(xlabel=r'$\theta_{error}$ (rad)', ylabel='PDF', normed=True, label_suffix='s',
                                       title=suff+' '+suff2)
                ax.set_xlim(0, np.pi)
                ax.set_ylim(0, 1.)
                plotter.save(fig)

    def compute_pulling_direction_variation_after_isolated_leading_attachments(self, redo, redo_hist=False):
        dtheta = .25
        bins = np.arange(0, np.pi + dtheta / 2., dtheta)
        # bins = np.array(list(-bins[::-1]) + list(bins))
        for suff in ['inside', 'outside']:

            result_name = 'pulling_direction_variation_after_isolated_%s_leading_attachment' % suff
            label = 'pulling direction variation after isolated %s leading_attachment' % suff
            description = 'Food Orientation difference between -2s and 0s and between 0s and 2s ' \
                          ' after isolated %s leading_attachment' % suff
            name_attachment = 'isolated_%s_leading_attachment_intervals' % suff
            self._get_pulling_direction_variation(result_name, name_attachment, label, description, redo)

            hist_name = self.compute_hist(result_name, bins=bins, redo=redo, redo_hist=redo_hist, error=True)
            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
            fig, ax = plotter.plot(xlabel='(rad)', ylabel='PDF', title='')
            plotter.save(fig)

    def _get_pulling_direction_variation(self, result_name, name_attachment, label, description, redo):
        if redo:
            name_error = 'mm1s_food_direction_error'
            init_frame_name = 'first_attachment_time_of_outside_ant'
            self.exp.load([name_attachment, name_error, init_frame_name])

            index = self.exp.get_df(name_attachment).reset_index().set_index([id_exp_name, id_frame_name]).index
            index = index.sort_values()
            index_names = [id_exp_name, id_frame_name]
            self.exp.add_new_empty_dataset(name=result_name, index_names=index_names,
                                           column_names=['before', 'after'], index_values=index,
                                           category=self.category, label=label, description=description)

            df_error0 = self.exp.get_df(name_error).copy()
            df_error0.reset_index(inplace=True)
            df_error0[id_frame_name] += 200
            df_error0.set_index([id_exp_name, id_frame_name], inplace=True)

            df_error1 = self.exp.get_df(name_error).copy()

            df_error2 = self.exp.get_df(name_error).copy()
            df_error2.reset_index(inplace=True)
            df_error2[id_frame_name] -= 200
            df_error2.set_index([id_exp_name, id_frame_name], inplace=True)

            df_error0 = df_error0.reindex(index)
            df_error1 = df_error1.reindex(index)
            df_error2 = df_error2.reindex(index)

            df_error_diff0 = Geo.angle_distance_df(df_error0, df_error1)
            df_error_diff1 = Geo.angle_distance_df(df_error1, df_error2)

            self.exp.get_df(result_name)['before'] = np.abs(np.around(np.c_[df_error_diff0[:]], 3))
            self.exp.get_df(result_name)['after'] = np.abs(np.around(np.c_[df_error_diff1[:]], 3))

            self.change_first_frame(result_name, init_frame_name)
            # df_res = self.exp.get_df(result_name).loc[pd.IndexSlice[:, 0:], :]
            # self.exp.change_df(result_name, df_res)

            self.exp.write(result_name)

    def compute_pulling_direction_variation_after_leading_attachments(self, redo, redo_hist=False):
        dtheta = np.pi/10.
        bins = np.arange(0, np.pi + dtheta / 2., dtheta)
        # bins = np.array(list(-bins[::-1]) + list(bins))
        for suff in ['inside', 'outside']:

            result_name = 'pulling_direction_variation_after_%s_leading_attachment' % suff
            label = 'pulling direction variation after %s leading_attachment' % suff
            description = 'Food Orientation difference between -2s and 0s and between 0s and 2s ' \
                          ' after %s leading_attachment' % suff
            name_attachment = '%s_leading_attachment_intervals' % suff
            self._get_pulling_direction_variation(result_name, name_attachment, label, description, redo)

            hist_name = self.compute_hist(result_name, bins=bins, redo=redo, redo_hist=redo_hist, error=True)
            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
            fig, ax = plotter.plot(xlabel='(rad)', ylabel='PDF', title='')
            plotter.save(fig)

    def compute_pulling_direction_variation_after_leading_attachments_evol(self, redo):

        name_error = 'mm1s_food_direction_error_variation_hist'
        for suff in ['outside', 'inside']:
            name = 'pulling_direction_variation_after_%s_leading_attachment' % suff
            result_name = name+'_hist_evol'

            dtheta = np.pi/10.
            bins = np.arange(0, np.pi+dtheta, dtheta)

            dx = 0.5
            start_frame_intervals = np.array(np.arange(-1, 2., dx)*60*100, dtype=int)
            end_frame_intervals = np.array(start_frame_intervals + dx*60*100, dtype=int)

            label = 'pulling direction after %s leading_attachment distribution over time (rad)' % suff
            description = 'Histogram over time of the pulling directions after %s leading_attachment' % suff

            if redo:
                self.exp.load(name)

                self.exp.operation(name, lambda a: np.abs(a))
                self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                          end_frame_intervals=end_frame_intervals, bins=bins,
                                          result_name=result_name+'_after', category=self.category,
                                          label=label, description=description, column_to_hist='after')
                self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                          end_frame_intervals=end_frame_intervals, bins=bins,
                                          result_name=result_name+'_before', category=self.category,
                                          label=label, description=description, column_to_hist='before')
                self.exp.remove_object(name)
                self.exp.write(result_name+'_after')
                self.exp.write(result_name+'_before')
            else:
                self.exp.load(result_name+'_before')
                self.exp.load(result_name+'_after')

            self.exp.load(name_error)
            plotter_error = Plotter(root=self.exp.root, obj=self.exp.get_data_object(name_error))
            columns = self.exp.get_columns(result_name+'_after')
            fig, ax = BasePlotters().create_plot(figsize=(10, 15), nrows=3, ncols=2)
            cs = {'before': 'k', 'after': 'w'}
            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name+'_after'))
            for k, column in enumerate(columns):
                i = int(k/2)
                j = k-i*2

                plotter_error.plot(preplot=(fig, ax[i, j]), normed=True, label='all', c='grey')
                for suff2 in ['before', 'after']:
                    plotter = Plotter(
                        root=self.exp.root, obj=self.exp.get_data_object(result_name+'_'+suff2), column_name=column)
                    plotter.plot(
                        preplot=(fig, ax[i, j]), normed=True, xlabel=r'$d\theta$ (rad)', ylabel='PDF',
                        label_suffix='s', title=column, label=suff2, c=cs[suff2])

                ax[i, j].set_xlim(0, np.pi)
                ax[i, j].set_ylim(0, 1.)
            plotter.save(fig, name=result_name)

    def compute_food_direction_error_high_variation(self, redo, redo_hist=False):
        name_result = 'time_distance_food_direction_error_high_variation_leading_attachment'
        label = 'Time distance food direction error significant variation and leading attachment'
        description = 'Time distance food direction error significant variation and leading attachment'

        bins = range(0, 50, 2)
        list_exps = list([12, 20, 40, 30, 42, 55])
        if redo:
            name_error = 'mm1s_food_direction_error'
            name_attachment = 'leading_attachment_intervals'
            init_frame_name = 'first_attachment_time_of_outside_ant'
            self.exp.load([name_error, name_attachment, init_frame_name])

            self.exp.change_df(name_error, self.exp.get_df(name_error).loc[list_exps, :])

            self.change_first_frame(name_error, init_frame_name)
            self.change_first_frame(name_attachment, init_frame_name)

            eps = 0.25
            dt_min = 100
            dt_max = 300
            dtheta_min = 1.
            a_min = dtheta_min/dt_max

            res = []

            def get_high_variation4each_gr(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                print(id_exp)
                attach_times = self.exp.get_df(name_attachment).loc[id_exp].index.get_level_values(id_frame_name)
                attach_times = np.array(attach_times)
                attach_times.sort()

                arr = df.reset_index().drop(columns=id_exp_name).dropna().values
                arr = arr[arr[:, 0] > 0, :]
                d_arr = Geo.angle_distance(arr[1:, 1], arr[:-1, 1])
                arr[:, 1] = arr[0, 1]
                arr[1:, 1] += np.cumsum(d_arr)

                r = rdp(arr, epsilon=eps)

                tab_dt = np.diff(r[:, 0])
                tab_dtheta = np.abs(np.diff(r[:, 1]))
                tab_a, _ = np.abs(Geo.get_line_tab(r[:-1, :], r[1:, :]))

                dt_small_enough = (tab_dt < dt_max)
                dt_big_enough = (tab_dt > dt_min)
                da_quick_enough = (tab_a > a_min)
                dtheta_big_enough = (tab_dtheta > dtheta_min)
                mask = np.where(dt_small_enough*dt_big_enough*da_quick_enough*dtheta_big_enough)[0]
                tab_t = r[mask, 0]

                for t in tab_t:
                    tab_dt = t-attach_times
                    tab_dt = tab_dt[tab_dt > 0]
                    if len(tab_dt) > 0:
                        dt = np.min(tab_dt)
                        res.append((id_exp, t, dt))

            self.exp.groupby(name_error, id_exp_name, get_high_variation4each_gr)

            res = np.array(res, dtype=int)
            self.exp.add_new_dataset_from_array(
                array=res, name=name_result, index_names=[id_exp_name, id_frame_name],
                column_names=['dt'], category=self.category, label=label, description=description
            )
            self.exp.operation(name_result, lambda x: x/100.)

            self.exp.write(name_result)
            self.exp.remove_object(name_error)

        hist_name = self.compute_hist(name=name_result, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot()
        plotter.save(fig)
