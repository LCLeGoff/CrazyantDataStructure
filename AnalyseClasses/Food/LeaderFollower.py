import pandas as pd
import numpy as np
import scipy.stats as scs
import pylab as pb
import sklearn.decomposition as decomp

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_frame_name, id_ant_name
from Tools.MiscellaneousTools.Geometry import angle_distance_df, angle_distance, angle_mean


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
        self._get_feature_from_exp_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_mm10_food_speed_features(self):
        variable_name = 'mm10_food_speed'
        label_name = 'Food speed'
        result_name = 'mm10_food_speed_leader_feature'

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_mm1s_food_speed_features(self):
        variable_name = 'mm1s_food_speed'
        label_name = 'Food speed'
        result_name = 'mm1s_food_speed_leader_feature'

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_food_orientation_features(self):
        variable_name = 'food_velocity_phi'
        label_name = 'Food orientation'
        result_name = 'food_orientation_leader_feature'

        # self.exp.load(variable_name)

        # def remove_modulo(df: pd.DataFrame):
        #     id_exp = df.index.get_level_values(id_exp_name)[0]
        #
        #     df2 = df.loc[id_exp, :].values
        #     df2 = angle_distance(df2[:-1], df2[1:])
        #     df2 = np.concatenate([[df.iloc[0]], df2])
        #
        #     df2 = np.nancumsum(df2)
        #     df[:] = np.c_[df2]
        #
        #     return df
        #
        # df_res = self.exp.groupby(variable_name, id_exp_name, remove_modulo)
        # self.exp.change_df(variable_name, df_res)

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_mm10_food_orientation_features(self):
        variable_name = 'mm10_food_velocity_phi'
        label_name = 'Food orientation'
        result_name = 'mm10_food_orientation_leader_feature'

        # self.exp.load(variable_name)
        #
        # def remove_modulo(df: pd.DataFrame):
        #     id_exp = df.index.get_level_values(id_exp_name)[0]
        #
        #     df2 = df.loc[id_exp, :].values
        #     df2 = angle_distance(df2[:-1], df2[1:])
        #     df2 = np.concatenate([[df.iloc[0]], df2])
        #
        #     df2 = np.nancumsum(df2)
        #     df[:] = np.c_[df2]
        #
        #     return df
        #
        # df_res = self.exp.groupby(variable_name, id_exp_name, remove_modulo)
        # self.exp.change_df(variable_name, df_res)

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_mm1s_food_orientation_features(self):
        variable_name = 'mm1s_food_velocity_phi'
        label_name = 'Food orientation'
        result_name = 'mm1s_food_orientation_leader_feature'

        # self.exp.load(variable_name)
        #
        # def remove_modulo(df: pd.DataFrame):
        #     id_exp = df.index.get_level_values(id_exp_name)[0]
        #
        #     df2 = df.loc[id_exp, :].values
        #     df2 = angle_distance(df2[:-1], df2[1:])
        #     df2 = np.concatenate([[df.iloc[0]], df2])
        #
        #     df2 = np.nancumsum(df2)
        #     df[:] = np.c_[df2]
        #
        #     return df
        #
        # df_res = self.exp.groupby(variable_name, id_exp_name, remove_modulo)
        # self.exp.change_df(variable_name, df_res)

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_food_orientation_speed_features(self):
        variable_name = 'food_orientation_speed'
        label_name = 'Food orientation speed'
        result_name = 'food_orientation_speed_leader_feature'

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_food_rotation_features(self):
        variable_name = 'food_rotation'
        label_name = 'Food rotation'
        result_name = 'food_rotation_leader_feature'

        self.exp.load(variable_name)

        def remove_modulo(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]

            df2 = df.loc[id_exp, :].values
            df2 = angle_distance(df2[:-1], df2[1:])
            df2 = np.concatenate([[df.iloc[0]], df2])

            df2 = np.nancumsum(df2)
            df[:] = np.c_[df2]

            return df

        df_res = self.exp.groupby(variable_name, id_exp_name, remove_modulo)
        self.exp.change_df(variable_name, df_res)

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_mm10_food_rotation_features(self):
        variable_name = 'mm10_food_rotation'
        label_name = 'Food rotation'
        result_name = 'mm10_food_rotation_leader_feature'

        self.exp.load(variable_name)

        def remove_modulo(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]

            df2 = df.loc[id_exp, :].values
            df2 = angle_distance(df2[:-1], df2[1:])
            df2 = np.concatenate([[df.iloc[0]], df2])

            df2 = np.nancumsum(df2)
            df[:] = np.c_[df2]

            return df

        df_res = self.exp.groupby(variable_name, id_exp_name, remove_modulo)
        self.exp.change_df(variable_name, df_res)

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_mm1s_food_rotation_features(self):
        variable_name = 'mm1s_food_rotation'
        label_name = 'Food rotation'
        result_name = 'mm1s_food_rotation_leader_feature'

        self.exp.load(variable_name)

        def remove_modulo(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]

            df2 = df.loc[id_exp, :].values
            df2 = angle_distance(df2[:-1], df2[1:])
            df2 = np.concatenate([[df.iloc[0]], df2])

            df2 = np.nancumsum(df2)
            df[:] = np.c_[df2]

            return df

        df_res = self.exp.groupby(variable_name, id_exp_name, remove_modulo)
        self.exp.change_df(variable_name, df_res)

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_nb_carriers_features(self):
        variable_name = 'nb_carriers'
        label_name = 'Carrier number'
        result_name = 'nb_carriers_leader_feature'

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_food_confidence_features(self):
        variable_name = 'w1s_food_path_efficiency'
        label_name = 'Food confidence'
        result_name = 'food_confidence_leader_feature'

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_food_accuracy_features(self):
        variable_name = 'food_direction_error'
        label_name = 'Food accuracy'
        result_name = 'food_accuracy_leader_feature'

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_ant_speed_features(self):
        variable_name = 'speed'
        label_name = 'Ant speed'
        result_name = 'ant_speed_leader_feature'

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_ant_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_attachment_angle_features(self):
        variable_name = 'foodVelocity_foodAntAttachmentVector_angle'
        label_name = 'Attachment angle'
        result_name = 'attachment_angle_leader_feature'

        # self.exp.load(variable_name)
        #
        # def remove_modulo(df: pd.DataFrame):
        #     id_exp = df.index.get_level_values(id_exp_name)[0]
        #     id_ant = df.index.get_level_values(id_ant_name)[0]
        #     print(id_exp, id_ant)
        #
        #     df2 = df.loc[id_exp, id_ant, :].values
        #     df2 = angle_distance(df2[:-1], df2[1:])
        #     df2 = np.concatenate([[df.iloc[0]], df2])
        #
        #     mask = np.where(np.isnan(df2))[0]
        #     df2 = np.nancumsum(df2)
        #     df2[mask] = np.nan
        #     df[:] = np.c_[df2]
        #
        #     return df
        #
        # df_res = self.exp.groupby(variable_name, [id_exp_name, id_ant_name], remove_modulo)
        # self.exp.change_df(variable_name, df_res)

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_ant_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_mm10_attachment_angle_features(self):
        variable_name = 'mm10_foodVelocity_foodAntAttachmentVector_angle'
        label_name = 'Attachment angle'
        result_name = 'mm10_attachment_angle_leader_feature'

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_ant_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_mm1s_attachment_angle_features(self):
        variable_name = 'mm1s_foodVelocity_foodAntAttachmentVector_angle'
        label_name = 'Attachment angle'
        result_name = 'mm1s_attachment_angle_leader_feature'

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_ant_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_attachment_angle_not_modulo_features(self):
        variable_name = 'foodVelocity_foodAntAttachmentVector_angle'
        label_name = 'Attachment angle (not modulo)'
        result_name = 'attachment_angle_not_modulo_leader_feature'

        self.exp.load(variable_name)

        def remove_modulo(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            id_ant = df.index.get_level_values(id_ant_name)[0]
            print(id_exp, id_ant)

            df2 = df.loc[id_exp, id_ant, :].values
            df2 = angle_distance(df2[:-1], df2[1:])
            df2 = np.concatenate([[df.iloc[0]], df2])

            mask = np.where(np.isnan(df2))[0]
            df2 = np.nancumsum(df2)
            df2[mask] = np.nan
            df[:] = np.c_[df2]

            return df

        df_res = self.exp.groupby(variable_name, [id_exp_name, id_ant_name], remove_modulo)
        self.exp.change_df(variable_name, df_res)

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_ant_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_body_orientation_alignment_features(self):
        variable_name = 'foodVelocity_AntBodyOrientation_angle'
        label_name = 'Ant body and food orientation alignment'
        result_name = 'body_orientation_alignment_leader_feature'

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_ant_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_body_rotation_alignment_features(self):
        variable_name = 'foodRotation_AntBodyOrientation_angle'
        label_name = 'Ant body and food rotation alignment'
        result_name = 'body_rotation_alignment_leader_feature'

        attachment_name = 'ant_attachment_intervals'
        self._get_feature_from_exp_ant_frame_indexed(variable_name, label_name, result_name, attachment_name)

    def prepare_outside_features(self):
        result_name = 'outside_leader_feature'
        label_name = 'From outside'

        variable_name = 'from_outside'
        attachment_name = 'ant_attachment_intervals'
        self.exp.load([variable_name, attachment_name])

        label = label_name + ' feature for leader/follower'
        description = label_name + ' prepared to be used as feature to discriminate leading/following attachments, ' \
                                   'based on '+variable_name

        index_values = self.exp.get_index(attachment_name)
        self.exp.add_new_empty_dataset(name=result_name, index_names=[id_exp_name, id_ant_name, id_frame_name],
                                       column_names=[result_name], index_values=index_values, fill_value=0,
                                       category=self.category, label=label, description=description)

        for id_exp, id_ant, from_outside in self.exp.get_df(variable_name).reset_index().values:
            try:
                self.exp.get_df(result_name).loc[id_exp, id_ant, :] = from_outside
            except KeyError:
                pass

        self.exp.write(result_name)

    def _get_feature_from_exp_frame_indexed(self, variable_name, label_name, result_name, attachment_name):

        label = label_name + ' feature for leader/follower'
        description = label_name + ' prepared to be used as feature to discriminate leading/following attachments, ' \
                                   'based on '+variable_name

        last_frame_name = 'food_exit_frames'
        self.exp.load([variable_name, attachment_name, last_frame_name, 'fps'], reload=False)

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

                    var_arr = np.array(var_df.reindex(time_intervals)[variable_name])

                    self.exp.get_df(result_name).loc[(id_exp, id_ant, attach_frame), :] = var_arr

        self.exp.groupby(variable_name, id_exp_name, func=get_variable4each_group)
        self.exp.write(result_name)

    def _get_feature_from_exp_ant_frame_indexed(self, variable_name, label_name, result_name, attachment_name):

        label = label_name + ' feature for leader/follower'
        description = label_name + ' prepared to be used as feature to discriminate leading/following attachments, ' \
                                   'based on '+variable_name

        last_frame_name = 'food_exit_frames'
        self.exp.load([variable_name, attachment_name, last_frame_name, 'fps'], reload=False)

        t0, t1, dt = -3, 3, 0.01
        time_intervals = np.around(np.arange(t0, t1 + dt, dt), 2)
        index_values = self.exp.get_index(attachment_name)
        self.exp.add_new_empty_dataset(name=result_name, index_names=[id_exp_name, id_ant_name, id_frame_name],
                                       column_names=np.array(time_intervals, dtype=str), index_values=index_values,
                                       category=self.category, label=label, description=description)

        for id_exp, id_ant, attach_frame in index_values:
            fps = self.exp.get_value('fps', id_exp)
            last_frame = self.exp.get_value(last_frame_name, id_exp)

            if attach_frame < last_frame:
                f0 = int(attach_frame + time_intervals[0] * fps)
                f1 = int(attach_frame + time_intervals[-1] * fps)

                var_df = self.exp.get_df(variable_name).loc[id_exp, id_ant, f0:f1].loc[id_exp, id_ant, :]
                var_df.index -= attach_frame
                var_df.index /= fps

                var_arr = np.array(var_df.reindex(time_intervals)[variable_name])

                self.exp.get_df(result_name).loc[(id_exp, id_ant, attach_frame), :] = var_arr

        self.exp.write(result_name)

    def draw_all_min_max(self):
        names = ['food_speed', 'food_confidence', 'food_orientation_speed',
                 'food_accuracy', 'food_orientation', 'food_rotation']
        for name in names:
            pb.subplots()
            feature_name = name + '_leader_feature'
            self.exp.load([feature_name])
            self.visualisation_after_vs_before(feature_name, title='Min Max ' + self.exp.get_label(feature_name),
                                               fct=self.apply_min_max)
        pb.show()

    def draw_all_min_max_diff(self):
        names = ['food_speed', 'food_orientation', 'food_orientation_speed',
                 'food_rotation', 'food_confidence', 'food_accuracy']
        for name in names:
            pb.subplots()
            feature_name = name + '_leader_feature'
            self.exp.load([feature_name])

            name_diff = self.get_differential(feature_name)
            self.visualisation_after_vs_before(name_diff, title='Min max ' + self.exp.get_label(feature_name) + ' diff',
                                               fct=self.apply_min_max)
        pb.show()

    def draw_all_std(self):
        names = ['food_speed', 'food_orientation', 'food_orientation_speed',
                 'food_rotation', 'food_confidence', 'food_accuracy']
        for name in names:
            pb.subplots()
            feature_name = name + '_leader_feature'
            self.exp.load([feature_name])
            self.visualisation_after_vs_before(
                feature_name, title='Std ' + self.exp.get_label(feature_name), fct=self.apply_std)
        pb.show()

    def draw_all_std_diff(self):
        names = ['food_speed', 'food_orientation', 'food_orientation_speed',
                 'food_rotation', 'food_confidence', 'food_accuracy']
        for name in names:
            pb.subplots()
            feature_name = name + '_leader_feature'
            self.exp.load([feature_name])

            name_diff = self.get_differential(feature_name)
            self.visualisation_after_vs_before(name_diff, title='Std ' + self.exp.get_label(feature_name) + ' diff',
                                               fct=self.apply_std)
        pb.show()

    def draw_all_means(self):
        names = ['food_speed', 'food_orientation', 'food_orientation_speed',
                 'food_rotation', 'food_confidence', 'food_accuracy']
        for name in names:
            pb.subplots()
            feature_name = name + '_leader_feature'
            self.exp.load([feature_name])
            self.visualisation_after_vs_before(feature_name,
                                               title='Mean ' + self.exp.get_label(feature_name), fct=self.apply_mean)
        pb.show()

    def draw_all_means_diff(self):
        names = ['food_speed', 'food_orientation', 'food_orientation_speed',
                 'food_rotation', 'food_confidence', 'food_accuracy']
        for name in names:
            pb.subplots()
            feature_name = name + '_leader_feature'
            self.exp.load([feature_name])

            name_diff = self.get_differential(feature_name)
            self.visualisation_after_vs_before(name_diff, title='Mean ' + self.exp.get_label(feature_name) + ' diff',
                                               fct=self.apply_mean)
        pb.show()

    def draw_all_std_vs_mean(self):
        names = ['food_speed', 'food_orientation', 'food_orientation_speed', 'food_rotation', 'ant_speed',
                 'attachment_angle', 'body_orientation_alignment', 'body_rotation_alignment',
                 'food_confidence', 'food_accuracy']
        for name in names:
            pb.subplots()
            feature_name = name + '_leader_feature'
            self.exp.load([feature_name])
            self.visualisation_std_vs_mean(
                feature_name, title='Mean vs std ' + self.exp.get_label(feature_name))
        pb.show()

    def draw_all_std_vs_mean_diff(self):
        names = ['food_speed', 'food_orientation', 'food_orientation_speed', 'food_rotation', 'ant_speed',
                 'attachment_angle', 'body_orientation_alignment', 'body_rotation_alignment',
                 'food_confidence', 'food_accuracy']
        for name in names:
            pb.subplots()
            feature_name = name + '_leader_feature'
            self.exp.load([feature_name])
            name_diff = self.get_differential(feature_name)
            self.visualisation_std_vs_mean(
                name_diff, title='Mean vs std ' + self.exp.get_label(feature_name) + ' diff')
        pb.show()

    def draw_3_main(self):
        names = ['food_speed', 'food_orientation_speed', 'food_rotation']
        self.exp.load([name+'_leader_feature' for name in names])

        # def fct(var_out_df, var_in_df):
        #
        #     before_outside = self.apply_min_max(var_out_df.loc[:, :0])
        #     after_outside = self.apply_min_max(var_out_df.loc[:, 0:])
        #     before_inside = self.apply_min_max(var_in_df.loc[:, :0])
        #     after_inside = self.apply_min_max(var_in_df.loc[:, 0:])
        #
        #     return (after_outside-before_outside).ravel(), (after_inside-before_inside).ravel()

        # def fct(var_out_df, var_in_df):
        #
        #     before_outside = self.apply_std(var_out_df.loc[:, :0])
        #     after_outside = self.apply_std(var_out_df.loc[:, 0:])
        #     before_inside = self.apply_std(var_in_df.loc[:, :0])
        #     after_inside = self.apply_std(var_in_df.loc[:, 0:])
        #
        #     return (after_outside-before_outside).ravel(), (after_inside-before_inside).ravel()

        def fct(var_df):

            # return self.apply_mean(var_out_df.loc[:, 0:]).ravel(), self.apply_mean(var_in_df.loc[:, 0:]).ravel()
            return self.apply_std(var_df.loc[:, 0:]).ravel(), self.apply_std(var_df.loc[:, 0:]).ravel()

        self.draw_features(names, fct)

    def test(self):
        name_speed = 'mm10_food_speed'
        name_attach_angle = 'mm10_attachment_angle'

        self.exp.load([name_speed+'_leader_feature', name_attach_angle+'_leader_feature'])

        def fct1(var_df):
            return self.apply_min_max(var_df.loc[:, 0:]).ravel()

        def fct2(var_df):
            return self.apply_mean_angle(var_df.loc[:, 0:]).ravel()

        self.draw_features([name_speed, name_attach_angle], [fct1, fct2])

        def fct1(var_df):
            return self.apply_min_max_angle(var_df.loc[:, 0:]).ravel()

        def fct2(var_df):
            return self.apply_mean_angle(var_df.loc[:, 0:]).ravel()
        self.exp.add_copy(name_attach_angle+'_leader_feature', name_attach_angle+'2_leader_feature')
        self.draw_features([name_attach_angle, name_attach_angle+'2'], [fct1, fct2])

    def draw_features(self, names, list_fct, bi_color=True):
        outside_name = 'outside_leader_feature'
        manual_name = 'manual_leading_attachments'
        if not(isinstance(list_fct, list)):
            list_fct = [list_fct for _ in range(len(names))]

        self.exp.load([outside_name, manual_name])

        if bi_color is True:
            df_color = self.exp.get_df(manual_name)[manual_name].copy()
            df_color.index = df_color.index.swaplevel(id_frame_name, id_ant_name)
            vmax = 1
        else:
            df_color = self.exp.get_df(manual_name).copy()
            df_color.reset_index(inplace=True)
            df_color.set_index([id_exp_name, id_frame_name], inplace=True)
            df_color = df_color.drop(columns=manual_name)
            df_color['f0'] = df_color.index.get_level_values(id_frame_name)

            def for_each_group(df_gr: pd.DataFrame):
                df_gr['f1'] = np.nan
                df_gr['f1'].iloc[:-1] = df_gr['f0'].iloc[1:].values
                return df_gr

            df_color = df_color.groupby(id_exp_name).apply(for_each_group)
            df_color['f1'] -= df_color['f0']
            df_color.reset_index(inplace=True)
            df_color.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)
            df_color = df_color.drop(columns=['f0'])
            df_leader = self.exp.get_df(manual_name)[manual_name].copy()
            df_leader.index = df_leader.index.swaplevel(id_frame_name, id_ant_name)
            df_color['f1'] *= 1 - df_leader
            vmax = 200

        manual_idx = df_color.dropna().index
        df_outside = self.exp.get_df(outside_name).copy()
        df_outside = df_outside.reindex(manual_idx)
        outside_idx = df_outside[df_outside == 1].dropna().index
        inside_idx = df_outside[df_outside == 0].dropna().index
        outside_color = df_color.loc[outside_idx].values.ravel()
        inside_color = df_color.loc[inside_idx].values.ravel()
        df_outside_features = pd.DataFrame(np.zeros((len(outside_idx), len(names))), index=outside_idx, columns=names)
        df_inside_features = pd.DataFrame(np.zeros((len(inside_idx), len(names))), index=inside_idx, columns=names)
        for i, name in enumerate(names):
            df = self.exp.get_df(name + '_leader_feature').copy()
            df = df.reindex(manual_idx)
            df.columns = df.columns.astype(float)

            var_outside_df = df.reindex(outside_idx)
            var_inside_df = df.reindex(inside_idx)

            df_outside_features[name] = list_fct[i](var_outside_df)
            df_inside_features[name] = list_fct[i](var_inside_df)
        cmap = 'hot'
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pb.subplots()
                pb.scatter(df_inside_features[names[i]], df_inside_features[names[j]],
                           marker='^', c=inside_color, label='inside', cmap=cmap, edgecolors='k', vmax=vmax)
                pb.scatter(df_outside_features[names[i]], df_outside_features[names[j]],
                           marker='o', c=outside_color, label='outside', cmap=cmap, edgecolors='k', vmax=vmax)
                pb.legend()
                pb.xlabel(names[i])
                pb.ylabel(names[j])
        pb.colorbar()
        pb.show()

    def get_differential(self, name):
        name_diff = name + '_diff'
        self.exp.add_copy(old_name=name, new_name=name_diff, replace=True)
        df = self.exp.get_df(name_diff).copy()
        df.columns = df.columns.copy().astype(float)+0.01

        self.exp.get_df(name_diff).columns = self.exp.get_df(name_diff).columns.astype(float)
        self.exp.get_data_object(name_diff).df -= df
        return name_diff

    @staticmethod
    def apply_mean(df):
        return df.mean(axis=1).values.ravel()

    @staticmethod
    def apply_mean_angle(df):
        df2 = np.exp(1j*df)
        df2 = df2.mean(axis=1)
        df2 = np.angle(df2)
        return df2

    @staticmethod
    def apply_std(df):
        return df.std(axis=1).values.ravel()

    @staticmethod
    def apply_min_max(df):
        return (df.max(axis=1)-df.min(axis=1)).values.ravel()

    @staticmethod
    def apply_min_max_angle(df):
        return (angle_distance_df(df.max(axis=1), df.min(axis=1))).values.ravel()

    def visualisation_after_vs_before(self, name, title, fct):
        outside_name = 'outside_leader_feature'
        manual_name = 'manual_leading_attachments'
        self.exp.load([outside_name, manual_name])

        df_color = self.exp.get_df(manual_name).copy()
        manual_idx = df_color.dropna().index.swaplevel(id_ant_name, id_frame_name)

        df = self.exp.get_df(name).copy()
        df = df.reindex(manual_idx)
        df.columns = df.columns.astype(float)

        df_outside = self.exp.get_df(outside_name).copy()
        df_outside = df_outside.reindex(manual_idx)

        outside_idx = df_outside[df_outside == 1].dropna().index
        var_outside_df = df.reindex(outside_idx)
        outside_color = df_color.loc[var_outside_df.index.swaplevel(id_ant_name, id_frame_name)].values.ravel()

        inside_idx = df_outside[df_outside == 0].dropna().index
        var_inside_df = df.reindex(inside_idx)
        inside_color = df_color.loc[var_inside_df.index.swaplevel(id_ant_name, id_frame_name)].values.ravel()

        before_outside = fct(var_outside_df.loc[:, :0])
        before_inside = fct(var_inside_df.loc[:, :0])
        after_outside = fct(var_outside_df.loc[:, 0:])
        after_inside = fct(var_inside_df.loc[:, 0:])

        cmap = 'bwr'
        pb.scatter(
            before_inside, after_inside, marker='^', c=inside_color, label='inside', cmap=cmap, edgecolors='k')
        pb.scatter(
            before_outside, after_outside, marker='o', c=outside_color, label='outside', cmap=cmap, edgecolors='k')
        pb.legend()
        pb.xlabel('before')
        pb.ylabel('after')
        pb.title(title)

    def visualisation_std_vs_mean(self, name, title):
        outside_name = 'outside_leader_feature'
        manual_name = 'manual_leading_attachments'
        self.exp.load([outside_name, manual_name])

        df_color = self.exp.get_df(manual_name).copy()
        manual_idx = df_color.dropna().index.swaplevel(id_ant_name, id_frame_name)

        df = self.exp.get_df(name).copy()
        df = df.reindex(manual_idx)
        df.columns = df.columns.astype(float)

        df_outside = self.exp.get_df(outside_name).copy()
        df_outside = df_outside.reindex(manual_idx)

        outside_idx = df_outside[df_outside == 1].dropna().index
        var_outside_df = df.reindex(outside_idx)
        outside_color = df_color.loc[var_outside_df.index.swaplevel(id_ant_name, id_frame_name)].values.ravel()

        inside_idx = df_outside[df_outside == 0].dropna().index
        var_inside_df = df.reindex(inside_idx)
        inside_color = df_color.loc[var_inside_df.index.swaplevel(id_ant_name, id_frame_name)].values.ravel()

        if name == 'ant_speed':
            mean_outside = self.apply_mean(var_outside_df.loc[:, :0])
            std_outside = self.apply_std(var_outside_df.loc[:, :0])
            mean_inside = self.apply_mean(var_inside_df.loc[:, :0])
            std_inside = self.apply_std(var_inside_df.loc[:, :0])
        else:
            mean_outside = self.apply_mean(var_outside_df.loc[:, 0:])
            std_outside = self.apply_std(var_outside_df.loc[:, 0:])
            mean_inside = self.apply_mean(var_inside_df.loc[:, 0:])
            std_inside = self.apply_std(var_inside_df.loc[:, 0:])

        cmap = 'bwr'
        pb.scatter(
            mean_inside, std_inside, marker='^', c=inside_color, label='inside', cmap=cmap, edgecolors='k')
        pb.scatter(
            mean_outside, std_outside, marker='o', c=outside_color, label='outside', cmap=cmap, edgecolors='k')
        pb.legend()
        pb.xlabel('mean')
        pb.ylabel('std')
        pb.title(title)

    def test2(self):
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

        name = names[0]
        before_leader = []
        after_leader = []
        before_follower = []
        after_follower = []
        self.exp.groupby(outside_name, [id_exp_name, id_frame_name], get_error4each_group)
        pb.plot(before_follower, after_follower, 'o', c='gray', label='outside follower')
        pb.plot(before_leader, after_leader, 'o', c='k', label='outside leader')

        before_leader = []
        after_leader = []
        before_follower = []
        after_follower = []
        self.exp.groupby(inside_name, [id_exp_name, id_frame_name], get_error4each_group)
        pb.plot(before_follower, after_follower, '*', c='r', label='inside follower')
        pb.plot(before_leader, after_leader, '*', c='orange', label='inside leader')

        pb.legend()
        pb.title(name)
        pb.show()

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
