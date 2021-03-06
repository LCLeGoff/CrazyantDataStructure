import pandas as pd
import numpy as np

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from DataStructure.VariableNames import id_frame_name, id_exp_name, id_ant_name
from ExperimentGroups import ExperimentGroups
from Scripts.root import root


class AnalyseClassDecorator:
    def __init__(self, group, exp: ExperimentGroups = None):
        if exp is None:
            self.exp = ExperimentGroupBuilder(root).build(group)
        else:
            self.exp = exp
        pass

    def compute_hist(self, name, bins, hist_name=None, error=False,
                     hist_label=None, hist_description=None, redo=False, redo_hist=False, write=True):
        if redo is True or redo_hist is True:
            self.exp.load(name)

            columns = self.exp.get_columns(name)
            if len(columns) == 1:
                hist_name = self.exp.hist1d(name_to_hist=name, result_name=hist_name, error=error,
                                            bins=bins, label=hist_label, description=hist_description)
            else:
                if hist_name is None:
                    hist_name = name + '_hist'
                for column in columns:
                    self.exp.hist1d(name_to_hist=name, result_name=column, error=False,
                                    column_to_hist=column, bins=bins,
                                    label=hist_label, description=hist_description)

                index = self.exp.get_index(columns[0])
                self.exp.add_new_empty_dataset(
                    name=hist_name, index_names=['bins'], column_names=columns,
                    index_values=index, category=self.exp.get_category(name),
                    label=self.exp.get_label(columns[0]), description=self.exp.get_description(columns[0]))

                for column in columns:
                    self.exp.get_df(hist_name)[column] = self.exp.get_df(column).values
                    self.exp.remove_object(column)

            if write:
                self.exp.write(hist_name)

        else:
            if hist_name is None:
                hist_name = name + '_hist'
            self.exp.load(hist_name)

        return hist_name

    def compute_surv(
            self, name, surv_name=None, hist_label=None, hist_description=None,
            redo=False, redo_hist=False, write=True):

        if redo is True or redo_hist is True:
            self.exp.load(name)

            columns = self.exp.get_columns(name)
            if len(columns) == 1:
                surv_name = self.exp.survival_curve(
                    name=name, result_name=surv_name, label=hist_label, description=hist_description)
            else:
                if surv_name is None:
                    surv_name = name + '_surv'
                for column in columns:
                    self.exp.survival_curve(
                        name=name, result_name=column, column_to_hist=column,
                        label=hist_label, description=hist_description)

                index = self.exp.get_index(columns[0])
                self.exp.add_new_empty_dataset(
                    name=surv_name, index_names=['bins'], column_names=columns,
                    index_values=index, category=self.exp.get_category(name),
                    label=self.exp.get_label(columns[0]), description=self.exp.get_description(columns[0]))

                for column in columns:
                    self.exp.get_df(surv_name)[column] = self.exp.get_df(column).values
                    self.exp.remove_object(column)

            if write:
                self.exp.write(surv_name)

        else:
            if surv_name is None:
                surv_name = name + '_surv'
            self.exp.load(surv_name)

        return surv_name

    def change_first_frame(self, name, frame_name):

        index_names = self.exp.get_index(name).names
        self.exp.load(frame_name)

        new_times = 'new_times'
        self.exp.add_copy1d(name_to_copy=name, copy_name=new_times, replace=True)
        self.exp.get_df(new_times).loc[:, new_times] = self.exp.get_index(new_times).get_level_values(id_frame_name)
        self.exp.operation_between_2names(
            name1=new_times, name2=frame_name, func=lambda a, b: a - b, col_name1=new_times)
        self.exp.get_df(new_times).reset_index(inplace=True)

        self.exp.get_df(name).reset_index(inplace=True)
        self.exp.get_df(name).loc[:, id_frame_name] = self.exp.get_df(new_times).loc[:, new_times]
        self.exp.get_df(name).set_index(index_names, inplace=True)

    def change_normalize_frame(self, name, frame_name):

        index_names = self.exp.get_index(name).names
        self.exp.load(frame_name)

        new_times = 'new_times'
        self.exp.add_copy1d(name_to_copy=name, copy_name=new_times, replace=True)

        def do4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]

            values = np.array(df.index.get_level_values(id_frame_name))
            mi = np.min(np.abs(values))
            ma = np.max(values)
            values = (values-mi)/(ma-mi)
            self.exp.get_df(new_times).loc[id_exp, :][new_times] = values

        self.exp.groupby(new_times, id_exp_name, do4each_group)

        self.exp.get_df(new_times).reset_index(inplace=True)

        self.exp.get_df(name).reset_index(inplace=True)
        self.exp.get_df(name).loc[:, id_frame_name] = self.exp.get_df(new_times).loc[:, new_times]
        self.exp.get_df(name).set_index(index_names, inplace=True)

    def change_first_frame2d(self, name, frame_name):

        index_names = self.exp.get_index(name).names
        self.exp.load(frame_name)

        new_times = 'new_times'
        self.exp.add_copy2d(name_to_copy=name, copy_name=new_times, replace=True)
        self.exp.get_df(new_times).loc[:, new_times] = self.exp.get_index(new_times).get_level_values(id_frame_name)
        self.exp.operation_between_2names(
            name1=new_times, name2=frame_name, func=lambda a, b: a - b, col_name1=new_times)
        self.exp.get_df(new_times).reset_index(inplace=True)

        self.exp.get_df(name).reset_index(inplace=True)
        self.exp.get_df(name).loc[:, id_frame_name] = self.exp.get_df(new_times).loc[:, new_times]
        self.exp.get_df(name).set_index(index_names, inplace=True)

    def cut_last_frames_for_indexed_by_exp_frame_indexed(self, name, frame_name):

        def cut4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            last_frame = self.exp.get_value(frame_name, id_exp)
            self.exp.get_df(name).loc[pd.IndexSlice[id_exp, last_frame:], :] = np.nan
            return df

        self.exp.groupby(name, id_exp_name, cut4each_group)
        self.exp.get_df(name).dropna(inplace=True)

    def cut_last_frames_for_indexed_by_exp_ant_frame_indexed(self, name, frame_name):

        def cut4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            last_frame = self.exp.get_value(frame_name, id_exp)
            self.exp.get_df(name).loc[id_exp, :, last_frame:] = np.nan
            return df

        self.exp.groupby(name, id_exp_name, cut4each_group)
        self.exp.get_df(name).dropna(inplace=True)

    def reindexing_exp_frame_indexed_by_exp_ant_frame_indexed(self, name_to_reindex, name2, column_names=None):
        if column_names is None:
            column_names = self.exp.get_columns(name_to_reindex)

        id_exps = self.exp.get_df(name2).index.get_level_values(id_exp_name)
        id_ants = self.exp.get_df(name2).index.get_level_values(id_ant_name)
        frames = self.exp.get_df(name2).index.get_level_values(id_frame_name)
        idxs = pd.MultiIndex.from_tuples(list(zip(id_exps, frames)), names=[id_exp_name, id_frame_name])

        df = self.exp.get_df(name_to_reindex).copy()
        df = df.reindex(idxs)
        df[id_ant_name] = id_ants
        df.reset_index(inplace=True)
        df.columns = [id_exp_name, id_frame_name]+column_names+[id_ant_name]
        df.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)

        return df
