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
        self.df = self.pandas_index_manager.concat_dfs(self.df, df)

        if replace:
            idx_to_keep = ~self.df.index.duplicated(keep='last')
            self.df = self.df[idx_to_keep]

    def mean_over_ants(self):
        return self.df.mean(level=['id_exp', 'id_ant'])

    def mean_over_experiments(self):
        return self.df.mean(level='id_exp')

    def mean_over_frames(self):
        return self.df.mean(level=['id_exp', 'id_ant', 'frame'])

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

    def fill_delta_df_without_filter(self, delta_df):

        index_dict = self.pandas_index_manager.get_dict_id_exp_ant_frame(self.df)

        for id_exp in index_dict:
            for id_ant in index_dict[id_exp]:
                frame_list = sorted(index_dict[id_exp][id_ant])
                for ii in range(len(frame_list) - 1):
                    frame0 = frame_list[ii]
                    frame1 = frame_list[ii + 1]
                    val0 = float(self.df.loc[(id_exp, id_ant, frame0)])
                    val1 = float(self.df.loc[(id_exp, id_ant, frame1)])
                    delta_df.loc[(id_exp, id_ant, frame0)] = val1 - val0
                delta_df.loc[(id_exp, id_ant, frame_list[-1])] = np.nan

    def fill_delta_df_with_filter(self, delta_df, filter_obj):

        index_dict = self.pandas_index_manager.get_dict_id_exp_ant_frame(self.df)

        for id_exp in index_dict:
            for id_ant in index_dict[id_exp]:
                exp_ant_delta_array = self.df.loc[pd.IndexSlice[id_exp, id_ant, :], :]
                exp_ant_delta_array = self.pandas_index_manager.convert_df_to_array(exp_ant_delta_array)

                exp_ant_filter_array = filter_obj.df.loc[pd.IndexSlice[id_exp, id_ant, :], :]
                exp_ant_filter_array = self.pandas_index_manager.convert_df_to_array(exp_ant_filter_array)

                idx_exp = np.where(exp_ant_delta_array[:, 0] == id_exp)[0]
                idx_ant = np.where(exp_ant_delta_array[:, 1] == id_ant)[0]

                interval_list = sorted(set(exp_ant_filter_array[:, -1]))

                for interval in interval_list:
                    idx_interval = np.where(exp_ant_filter_array[:, -1] == interval)[0]
                    idx_exp_ant_interval = np.intersect1d(np.intersect1d(idx_exp, idx_ant), idx_interval)
                    exp_ant_interval_delta_array = exp_ant_delta_array[idx_exp_ant_interval, :]
                    delta = exp_ant_interval_delta_array[:-1, -1] - exp_ant_interval_delta_array[1:, -1]
                    exp_ant_interval_delta_array[:-1, -1] = delta
                    exp_ant_interval_delta_array[-1, -1] = np.nan

                    idx_list = list(map(tuple, exp_ant_interval_delta_array[:, :-1].astype(int)))
                    for i, idx in enumerate(idx_list):
                        delta_df.loc[idx] = exp_ant_interval_delta_array[i, -1]
        self.pandas_index_manager.index_as_type_int(delta_df)

    # TODO: precise filter_level: exp, ant or None
    def fill_delta_df(self, delta_df, filter_obj):
        if filter_obj is None:
            self.fill_delta_df_without_filter(delta_df)
        else:
            self.fill_delta_df_with_filter(delta_df, filter_obj)
