import numpy as np
import pandas as pd

from DataStructure.VariableNames import id_exp_name, id_frame_name, id_ant_name

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from Tools.Plotter.Plotter import Plotter


class AnalyseLeadingAttachments(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'LeadingAttachment'

    def get_leading_attachment_intervals(self):
        leader_name = 'is_leader'
        self.exp.load(leader_name)
        for sup in ['', 'outside_', 'inside_']:
            name = '%sattachment_intervals' % sup
            self.exp.load(name)
            result_name = '%sleading_attachment_intervals' % sup

            self.exp.add_copy(old_name=name, new_name=result_name, category=self.category,
                              label='%s leading attachments', description='%s leading attachments')

            for id_exp, id_ant, frame in self.exp.get_index(name):
                if self.exp.get_index(leader_name).isin([(id_exp, id_ant, frame)]).any():
                    leading = self.exp.get_value(leader_name, (id_exp, id_ant, frame))
                    if leading == 0:
                        if self.exp.get_index(result_name).isin([(id_exp, id_ant, frame)]).any():
                            self.exp.get_df(result_name).loc[id_exp, id_ant, frame] = np.nan

            self.exp.change_df(result_name, self.exp.get_df(result_name).dropna())

            self.exp.write(result_name)

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

        dx = 0.01
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(0, 4., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of leading %s attachments in a 10s period over time'
        description = 'Number of leading %s attaching to the food in a 10s period over time'

        self._get_nb_attachments_evol(name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                      label % typ, description % typ, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Number of attachments',
                               label_suffix='s', marker='')
        plotter.plot_smooth(preplot=(fig, ax), window=50, c='orange', label='mean smoothed')
        plotter.save(fig)

    def compute_nb_leading_attachments_evol_around_first_outside_attachment(self, redo=False):

        typ = ''
        name = 'leading_attachment_intervals'

        result_name = 'nb_leading_attachments_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.01
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(-1, 3., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of leading %s attachments in a 10s period over time'
        description = 'Number of leading %s attaching to the food in a 10s period over time'

        self._get_nb_attachments_evol(name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                      label % typ, description % typ, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Mean of the number of attachments', label='Mean',
                               label_suffix='s', marker='', title='')
        # plotter.plot_fit(preplot=(fig, ax), typ='linear', window=[0, 95], label='linear fit')
        plotter.plot_fit(preplot=(fig, ax), typ='log', window=[95, 500], label='log fit')
        plotter.draw_legend(ax)
        plotter.draw_vertical_line(ax)
        plotter.save(fig)

        # plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        # fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Number of attachments',
        #                        label_suffix='s', marker='', display_legend=False)
        # plotter.plot_smooth(preplot=(fig, ax), window=50, c='orange', label='mean smoothed', display_legend=False)
        # plotter.draw_vertical_line(ax)
        # plotter.save(fig)

    def compute_nb_leading_outside_attachments_evol(self, redo=False):

        typ = 'outside'

        name = 'outside_leading_attachment_intervals'
        result_name = 'nb_outside_leading_attachments_evol'
        init_frame_name = 'food_first_frame'

        dx = 0.01
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(0, 4., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of leading %s attachments in a 10s period over time'
        description = 'Number of leading %s attaching to the food in a 10s period over time'

        self._get_nb_attachments_evol(name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                      label % typ, description % typ, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Number of attachments',
                               label_suffix='s', marker='')
        plotter.plot_smooth(preplot=(fig, ax), window=50, c='orange', label='mean smoothed')
        plotter.save(fig)

    def compute_nb_leading_outside_attachments_evol_around_first_outside_attachment(self, redo=False):

        typ = 'outside'
        name = 'outside_leading_attachment_intervals'

        result_name = 'nb_outside_leading_attachments_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.01
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(0, 3., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of leading %s attachments in a 10s period over time'
        description = 'Number of leading %s attaching to the food in a 10s period over time'

        self._get_nb_attachments_evol(name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                      label % typ, description % typ, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Mean of the number of attachments', label='Mean',
                               label_suffix='s', marker='', title='')
        # plotter.plot_fit(preplot=(fig, ax), typ='linear', window=[0, 550], label='linear fit', c='0.2')
        plotter.plot_fit(preplot=(fig, ax), typ='cst', window=[0, 550], label='cst fit')
        plotter.draw_legend(ax)
        # ax.set_ylim((0, 6))
        plotter.save(fig)

    def compute_nb_leading_inside_attachments_evol_around_first_outside_attachment(self, redo=False):

        typ = 'inside'
        name = 'inside_leading_attachment_intervals'

        result_name = 'nb_inside_leading_attachments_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.01
        dx2 = 1 / 6.
        start_frame_intervals = np.arange(-1, 3., dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of leading %s attachments in a 10s period over time'
        description = 'Number of leading %s attaching to the food in a 10s period over time'

        self._get_nb_attachments_evol(name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                      label % typ, description % typ, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Mean of the number of attachments', label='Mean',
                               label_suffix='s', marker='', title='')
        plotter.plot_fit(preplot=(fig, ax), typ='log', window=[95, 500], label='log fit')
        plotter.draw_legend(ax)
        plotter.draw_vertical_line(ax)
        plotter.save(fig)

    def _get_nb_attachments_evol(self, name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                 label, description, redo):
        if redo:

            self.exp.load(name)
            self.exp.load('food_x')
            self.change_first_frame(name, init_frame_name)
            self.change_first_frame('food_x', init_frame_name)

            last_frame_name = 'food_exit_frames'
            self.exp.load(last_frame_name)
            self.cut_last_frames_for_indexed_by_exp_frame_indexed(name, last_frame_name)

            x = (end_frame_intervals + start_frame_intervals) / 2. / 100.
            y = np.zeros(len(start_frame_intervals))
            for i in range(len(start_frame_intervals)):
                frame0 = int(start_frame_intervals[i])
                frame1 = int(end_frame_intervals[i])

                df = self.exp.get_df(name).loc[pd.IndexSlice[:, :, frame0:frame1], :]
                # n_exp = len(set(df_food.index.get_level_values(id_exp_name)))
                # if n_exp != 0:
                #     y[i] = len(df) / float(n_exp)
                df_food = self.exp.get_df('food_x').loc[pd.IndexSlice[:, frame0:frame1], :]
                y[i] = len(df) / len(df_food)*1000
            df = pd.DataFrame(y, index=x)
            self.exp.add_new_dataset_from_df(df=df, name=result_name, category=self.category,
                                             label=label, description=description)
            self.exp.write(result_name)
            self.exp.remove_object(name)

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

