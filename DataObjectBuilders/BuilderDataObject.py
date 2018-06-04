import pandas as pd
import numpy as np

from PandasIndexManager.PandasIndexManager import PandasIndexManager


class BuilderDataObject:
    def __init__(self, df):
        self.df = df

    def operation(self, fct):
        self.df = fct(self.df)

    def print(self, short=True):
        if short:
            print(self.df.head())
        else:
            print(self.df)

    def convert_df_to_array(self):
        return np.array(self.df.reset_index())

    def get_row(self, idx):
        return self.df.loc[idx, :]

    def get_row_of_idx_array(self, idx_array):
        return self.df.loc[list(map(tuple, np.array(idx_array))), :]

    def get_index_array_of_id_exp(self):
        return PandasIndexManager().get_array_id_exp(self.df)

    def get_index_array_of_id_exp_ant(self):
        return PandasIndexManager().get_array_id_exp_ant(self.df)

    def get_index_array_of_id_exp_ant_frame(self):
        return PandasIndexManager().get_array_id_exp_ant_frame(self.df)

    def get_index_dict_of_id_exp_ant(self):
        return PandasIndexManager().get_dict_id_exp_ant(self.df)

    def get_index_dict_of_id_exp_ant_frame(self):
        return PandasIndexManager().get_dict_id_exp_ant_frame(self.df)

    def get_array_of_all_ants_of_exp(self, id_exp):
        id_exp_ant_frame_array = self.get_index_array_of_id_exp_ant_frame()
        idx_where_id_exp = np.where(id_exp_ant_frame_array[:, 0] == id_exp)
        res = set(id_exp_ant_frame_array[idx_where_id_exp, 1])
        res = sorted(res)
        return np.array(res)

    def get_array_of_all_frames_of_exp(self, id_exp):
        id_exp_ant_frame_array = self.get_index_array_of_id_exp_ant_frame()
        idx_where_id_exp = np.where(id_exp_ant_frame_array[:, 0] == id_exp)[0]
        res = set(id_exp_ant_frame_array[idx_where_id_exp, 2])
        res = sorted(res)
        return np.array(res)

    def add_row(self, idx, value, replace=False):
        if replace is False and idx in self.df.index:
            raise IndexError('Index ' + str(idx) + ' already exists')
        else:
            self.df.loc[idx] = value

    def add_rows(self, idx_list, value_list, replace=False):
        if len(idx_list) == len(value_list):
            for ii in range(len(idx_list)):
                self.add_row(idx_list[ii], value_list[ii], replace=replace)
        else:
            raise IndexError('Index and value list not same lengths')

    def add_df_as_rows(self, df, replace=True):
        df.columns = self.df.columns
        dfs_to_concat = [self.df, df]
        self.df = pd.concat(dfs_to_concat)
        if replace:
            idx_to_keep = ~self.df.index.duplicated(keep='last')
            self.df = self.df[idx_to_keep]

    def mean_over_ants(self):
        return self.df.mean(level=['id_exp', 'id_ant'])

    def mean_over_experiments(self):
        return self.df.mean(level='id_exp')

    def mean_over_frames(self):
        return self.df.mean(level=['id_exp', 'id_ant', 'frame'])

    def mean_over(self, level_name):
        return self.df.mean(level=['id_exp', 'id_ant', 'frame', level_name])
