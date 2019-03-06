import pandas as pd
import numpy as np

from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class BuilderDataObject:
    def __init__(self, df):
        self.df = df
        self.pandas_index_manager = PandasIndexManager()

    def operation(self, fct):
        self.df = fct(self.df)

    def operation_with_data_obj(self, obj, fct, self_name_col=None, obj_name_col=None):
        if self_name_col is None:
            self_name_col = self.df.columns[0]
        if obj_name_col is None:
            obj_name_col = obj.name_col

        self.df[self_name_col] = fct(self.df[self_name_col], obj.df[obj_name_col])

    def print(self, short=True):
        if short:
            print(self.df.head())
        else:
            print(self.df)

    def convert_df_to_array(self):
        return self.pandas_index_manager.convert_df_to_array(self.df)

    def get_row(self, idx):
        return self.df.loc[idx, :]

    def get_row_of_idx_array(self, idx_array):
        return self.df.loc[list(map(tuple, np.array(idx_array))), :]

    def get_index_array_of_id_exp(self):
        return PandasIndexManager().get_array_id_exp(self.df)

    def get_index_array_of_id_exp_ant(self):
        return PandasIndexManager().get_array_id_exp_ant(self.df)

    def get_index_array_of_id_exp_frame(self):
        return PandasIndexManager().get_array_id_exp_frame(self.df)

    def get_index_array_of_id_exp_ant_frame(self):
        return PandasIndexManager().get_array_id_exp_ant_frame(self.df)

    def get_index_dict_of_id_exp_ant(self):
        return PandasIndexManager().get_dict_id_exp_ant(self.df)

    def get_index_dict_of_id_exp_frame(self):
        return PandasIndexManager().get_dict_id_exp_frame(self.df)

    def get_index_dict_of_id_exp_ant_frame(self):
        return PandasIndexManager().get_dict_id_exp_ant_frame(self.df)

    def get_array_of_all_ants_of_exp(self, id_exp):
        id_exp_ant_frame_array = self.get_index_array_of_id_exp_ant_frame()
        idx_where_id_exp = np.where(id_exp_ant_frame_array[:, 0] == id_exp)
        res = set(id_exp_ant_frame_array[idx_where_id_exp, 1])
        res = sorted(res)
        return np.array(res)

    def get_array_of_all_frames_of_exp(self, id_exp):
        if 'id_ant' not in self.df.columns:
            id_exp_frame_array = self.get_index_array_of_id_exp_ant_frame()
        else:
            id_exp_frame_array = self.get_index_array_of_id_exp_frame()
        idx_where_id_exp = np.where(id_exp_frame_array[:, 0] == id_exp)[0]
        res = set(id_exp_frame_array[idx_where_id_exp, 2])
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
        self.df = self.pandas_index_manager.concat_dfs(self.df, df)

        if replace:
            idx_to_keep = ~self.df.index.duplicated(keep='last')
            self.df = self.df[idx_to_keep]

    def mean_over_ants(self):
        return self.df.mean(level=['id_exp', 'id_ant'])

    def mean_over_experiments(self):
        return self.df.mean(level='id_exp')

    def mean_over_frames(self):
        if 'id_ant' not in self.df.columns:
            return self.df.mean(level=['id_exp', 'id_ant', 'frame'])
        else:
            return self.df.mean(level=['id_exp', 'frame'])

    def mean_over(self, level_df, mean_level=None, new_level_as=None):
        df = self.df.copy()
        filter_idx = 'new_idx'
        self.pandas_index_manager.add_index_level(
            df, filter_idx, np.array(level_df)
        )
        if mean_level is None:
            df = df.mean(level=[filter_idx])
        elif mean_level == 'exp':
            df = df.mean(level=['id_exp', filter_idx])
        elif mean_level == 'ant':
            df = df.mean(level=['id_exp', 'id_ant', filter_idx])
        else:
            raise ValueError(mean_level + ' not understood')
        if new_level_as is None:
            self.pandas_index_manager.remove_index_level(df, filter_idx)
        else:
            self.pandas_index_manager.rename_index_level(df, filter_idx, new_level_as)
        return df
