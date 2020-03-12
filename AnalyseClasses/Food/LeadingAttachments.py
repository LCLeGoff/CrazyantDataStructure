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

        dx = 0.05
        dx2 = 1.
        start_frame_intervals = np.arange(0, 2.5, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of leading %s attachments in a 10s period over time'
        description = 'Number of leading %s attaching to the food in a 10s period over time'

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

        label = 'Number of leading %s attachments in a 10s period over time'
        description = 'Number of leading %s attaching to the food in a 10s period over time'

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

        label = 'Number of leading %s attachments in a 10s period over time'
        description = 'Number of leading %s attaching to the food in a 10s period over time'

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
        dx2 = 6/6.
        start_frame_intervals = np.arange(0, 4, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of leading %s attachments in a 1s period over time'
        description = 'Number of leading %s attaching to the food in a 1s period over time'

        self._get_nb_attachments_evol(name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                      label % typ, description % typ, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot_with_error(xlabel='Time (s)', ylabel='Attachment probability (for 1s)',
                                          label='probability', marker='', title='')
        # plotter.plot_fit(preplot=(fig, ax), typ='linear', window=[0, 550], label='linear fit', c='0.2')
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name='p')
        plotter.plot_fit(preplot=(fig, ax), typ='cst', window=[0, 30], label='cst fit')
        ax.set_ylim((0.0, 0.2))
        plotter.draw_legend(ax)
        plotter.save(fig)

    def compute_nb_leading_inside_attachments_evol_around_first_outside_attachment(self, redo=False):

        typ = 'inside'
        name = 'inside_leading_attachment_intervals'

        result_name = 'nb_inside_leading_attachments_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.05
        dx2 = 1.
        start_frame_intervals = np.arange(-1, 2.5, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        label = 'Number of leading %s attachments in a 10s period over time'
        description = 'Number of leading %s attaching to the food in a 10s period over time'

        self._get_nb_attachments_evol(name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                      label % typ, description % typ, redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot_with_error(xlabel='Time (s)', ylabel='Attachment probability (for 1s)',
                                          label='probability', marker='', title='')
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name='p')
        plotter.plot_fit(preplot=(fig, ax), typ='cst')

        plotter.save(fig)

    def _get_nb_attachments_evol(self, name, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
                                 label, description, redo):
        if redo:

            self.exp.load([name, 'food_x', 'food_exit_frames'])

            self.cut_last_frames_for_indexed_by_exp_frame_indexed('food_x', 'food_exit_frames')
            self.cut_last_frames_for_indexed_by_exp_frame_indexed(name, 'food_exit_frames')

            self.change_first_frame(name, init_frame_name)
            self.change_first_frame('food_x', init_frame_name)

            list_exp = [3, 12, 20, 30, 42, 45, 47, 49, 55]

            x = start_frame_intervals / 2. / 100.
            y = np.full((len(start_frame_intervals), 3), np.nan)

            for i in range(len(start_frame_intervals)):
                frame0 = int(start_frame_intervals[i])
                frame1 = int(end_frame_intervals[i])

                df = self.exp.get_df(name).loc[pd.IndexSlice[list_exp, :, frame0:frame1], :]
                df_food = self.exp.get_df('food_x').loc[pd.IndexSlice[list_exp, frame0:frame1], :]

                n = round(len(df_food)/100)
                if n != 0:
                    p = len(df) / n
                    y[i, 0] = p
                    y[i, 1] = 1.96*np.sqrt(p*(1-p)/n)
                    y[i, 2] = 1.96*np.sqrt(p*(1-p)/n)

            mask = np.where(~np.isnan(y[:, 0]))[0]

            df = pd.DataFrame(y[mask, :], index=x[mask], columns=['p', 'err1', 'err2'])
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

