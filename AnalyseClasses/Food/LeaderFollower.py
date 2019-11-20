import pandas as pd
import numpy as np
import scipy.stats as scs
import pylab as pb
import sklearn.decomposition as decomp

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_frame_name, id_ant_name


class AnalyseLeaderFollower(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'LeaderFollower'
        self.exp.data_manager.create_new_category(self.category)

    def get_manual_leader_follower(self):

        result_name = 'manual_leading_attachments'
        outside_result_name = 'outside_manual_leading_attachments'
        inside_result_name = 'inside_manual_leading_attachments'

        label = 'Manual leading %s attachments'
        description = '1 (resp. 0) means the %s attachment had (resp. not) an influence on the object trajectory.' \
                      ' The data has been manually collected'

        df_all = pd.read_csv(self.exp.root + 'manual_leader_follower.csv')
        df_all.set_index([id_exp_name, id_frame_name, id_ant_name], inplace=True)
        df_all.columns = [result_name]

        self.exp.add_new_dataset_from_df(df=df_all, name=result_name, category=self.category,
                                         label=label % '', description=description % '')

        self.exp.load('from_outside')

        df_outside = df_all.copy()
        df_outside.columns = [outside_result_name]
        df_inside = df_all.copy()
        df_inside.columns = [inside_result_name]

        def split_out_inside4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            id_ant = df.index.get_level_values(id_ant_name)[0]
            from_outside = self.exp.get_value('from_outside', (id_exp, id_ant))

            if from_outside == 1:
                df_inside.loc[id_exp, :, id_ant] = np.nan
            else:
                df_outside.loc[id_exp, :, id_ant] = np.nan

        df_all.groupby([id_exp_name, id_ant_name]).apply(split_out_inside4each_group)
        df_inside.dropna(inplace=True)
        df_outside.dropna(inplace=True)

        self.exp.add_new_dataset_from_df(df=df_outside, name=outside_result_name, category=self.category,
                                         label=label % 'outside', description=description % 'outside')

        self.exp.add_new_dataset_from_df(df=df_inside, name=inside_result_name, category=self.category,
                                         label=label % 'inside', description=description % 'inside')

        self.exp.write([result_name, outside_result_name, inside_result_name])

    def print_manual_leader_stats(self):

        all_name = 'manual_leading_attachments'
        outside_name = 'outside_manual_leading_attachments'
        inside_name = 'inside_manual_leading_attachments'
        self.exp.load([all_name, outside_name, inside_name])

        print()
        for label, name in [('all', all_name), ('outside', outside_name), ('inside', inside_name)]:

            df = self.exp.get_df(name).dropna()
            m = float(np.nanmean(df))
            n = len(df)
            s1 = int(np.nansum(df))
            s2 = int(np.nansum(1-df))
            lower = scs.beta.ppf(0.025, s1, s2)
            upper = scs.beta.ppf(0.975, s1, s2)

            print('%s: %.3f, (%.3f, %.3f), %i' % (label, round(m, 3), round(lower, 3), round(upper, 3), n))

    def prepare_food_speed_features(self):
        variable_name = 'food_speed'
        label_name = 'Food speed'
        result_name = 'food_speed_leader_feature'

        attachment_name = 'ant_attachment_intervals'
        self._get_feature(variable_name, label_name, result_name, attachment_name)

    def prepare_food_orientation_features(self):
        variable_name = 'food_velocity_phi'
        label_name = 'Food orientation'
        result_name = 'food_orientation_leader_feature'

        attachment_name = 'ant_attachment_intervals'
        self._get_feature(variable_name, label_name, result_name, attachment_name)

    def prepare_food_rotation_features(self):
        variable_name = 'food_rotation'
        label_name = 'Food rotation'
        attachment_name = 'ant_attachment_intervals'
        result_name = variable_name+'_leader_feature'

        self._get_feature(variable_name, label_name, result_name, attachment_name)

    def _get_feature(self, variable_name, label_name, result_name, attachment_name):

        label = label_name + ' feature for leader/follower'
        description = label_name + ' prepared to be used as feature to discriminate leading/following attachments, ' \
                                   'based on '+variable_name

        last_frame_name = 'food_exit_frames'
        self.exp.load([variable_name, attachment_name, last_frame_name, 'fps'])

        t0, t1, dt = -3, 3, 0.01
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 2)
        index_values = self.exp.get_index(attachment_name)
        self.exp.add_new_empty_dataset(name=result_name, index_names=[id_exp_name, id_ant_name, id_frame_name],
                                       column_names=np.array(time_intervals, dtype=str), index_values=index_values,
                                       category=self.category, label=label, description=description)

        def get_variable4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            fps = self.exp.get_value('fps', id_exp)
            last_frame = self.exp.get_value(last_frame_name, id_exp)

            idx = self.exp.get_df(attachment_name).loc[id_exp, :].index

            for id_ant, attach_frame in idx:
                if attach_frame < last_frame:
                    f0 = int(attach_frame + time_intervals[0] * fps)
                    f1 = int(attach_frame + time_intervals[-1] * fps)

                    var_df = df.loc[pd.IndexSlice[id_exp, f0:f1], :]
                    var_df = var_df.loc[id_exp, :]
                    var_df.index -= attach_frame
                    var_df.index /= fps

                    var_df = var_df.reindex(time_intervals)

                    self.exp.get_df(result_name).loc[(id_exp, id_ant, attach_frame), :] = np.array(var_df[variable_name])

        self.exp.groupby(variable_name, id_exp_name, func=get_variable4each_group)
        self.exp.write(result_name)

    def test(self):
        names = ['food_direction_error', 'food_speed', 'food_velocity_phi', 'food_rotation',
                 'w2s_food_path_efficiency', 'food_angular_acceleration']

        all_name = 'manual_leading_attachments'
        outside_name = 'outside_manual_leading_attachments'
        inside_name = 'inside_manual_leading_attachments'
        self.exp.load(names+[all_name, outside_name, inside_name])

        def get_error4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            frame = df.index.get_level_values(id_frame_name)[0]
            leader = float(df.iloc[0])
            if ~np.isnan(leader):
                # print(id_exp, frame)

                m_before = np.nanmean(self.exp.get_df(name).loc[pd.IndexSlice[id_exp, frame - 300:frame], :])
                m_before = np.around(m_before, 6)
                m_after = np.nanmean(self.exp.get_df(name).loc[pd.IndexSlice[id_exp, frame:frame + 300], :])
                m_after = np.around(m_after, 6)
                if leader == 1:
                    before_leader.append(m_before)
                    after_leader.append(m_after)
                else:
                    before_follower.append(m_before)
                    after_follower.append(m_after)

        # name = names[0]
        # before_leader = []
        # after_leader = []
        # before_follower = []
        # after_follower = []
        # self.exp.groupby(outside_name, [id_exp_name, id_frame_name], get_error4each_group)
        # pb.plot(before_follower, after_follower, 'o', c='gray', label='outside follower')
        # pb.plot(before_leader, after_leader, 'o', c='k', label='outside leader')
        #
        #
        # before_leader = []
        # after_leader = []
        # before_follower = []
        # after_follower = []
        # self.exp.groupby(inside_name, [id_exp_name, id_frame_name], get_error4each_group)
        # pb.plot(before_follower, after_follower, '*', c='r', label='inside follower')
        # pb.plot(before_leader, after_leader, '*', c='orange', label='inside leader')

        # pb.legend()
        # pb.title(name)
        # pb.show()

        lg = len(self.exp.get_df(outside_name))
        nb_outside_leader = int(np.sum(self.exp.get_df(outside_name)))
        x_outside = np.zeros((lg, len(names)))
        y_outside = np.zeros(lg)
        y_outside[:nb_outside_leader] = 1

        lg = len(self.exp.get_df(inside_name))
        nb_inside_leader = int(np.sum(self.exp.get_df(inside_name)))
        x_inside = np.zeros((lg, len(names)))
        y_inside = np.zeros(lg)
        y_inside[:nb_inside_leader] = 1

        for i, name in enumerate(names):
            before_leader = []
            after_leader = []
            before_follower = []
            after_follower = []
            self.exp.groupby(outside_name, [id_exp_name, id_frame_name], get_error4each_group)
            x_outside[:nb_outside_leader, i] = np.array(after_leader)-np.array(before_leader)
            x_outside[nb_outside_leader:, i] = np.array(after_follower)-np.array(before_follower)

            before_leader = []
            after_leader = []
            before_follower = []
            after_follower = []
            self.exp.groupby(inside_name, [id_exp_name, id_frame_name], get_error4each_group)
            x_inside[:nb_inside_leader, i] = np.array(after_leader)-np.array(before_leader)
            x_inside[nb_inside_leader:, i] = np.array(after_follower)-np.array(before_follower)

        i_list = []
        for i in range(len(x_outside)):
            if not(any(np.isnan(x_outside[i, :]))):
                i_list.append(i)
        x_outside = x_outside[i_list, :]
        y_outside = y_outside[i_list]

        i_list = []
        for i in range(len(x_inside)):
            if not(any(np.isnan(x_inside[i, :]))):
                i_list.append(i)
        x_inside = x_inside[i_list, :]
        y_inside = y_inside[i_list]

        pca = decomp.PCA(n_components=2)
        pca.fit(x_outside)
        x_outside_pca = pca.transform(x_outside)
        pb.scatter(x_outside_pca[:, 0], x_outside_pca[:, 1], c=y_outside)

        pca = decomp.PCA(n_components=2)
        pca.fit(x_inside)
        x_inside_pca = pca.transform(x_inside)
        pb.scatter(x_inside_pca[:, 0], x_inside_pca[:, 1], c=y_inside, marker='*')

        pb.show()
