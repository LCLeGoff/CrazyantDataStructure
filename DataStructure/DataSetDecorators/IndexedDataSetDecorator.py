import numpy as np
import pandas as pd

from DataStructure.VariableNames import id_frame_name
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class IndexedDataSetDecorator:
    def __init__(self, df):
        self.df = df

    def rename_df(self, names):
        if isinstance(names, str):
            names = [names]
        self.df.columns = names

    def get_index_names(self):
        return self.df.index.names

    def get_column_names(self):
        return self.df.columns

    def get_nbr_index(self):
        return len(self.df.index.names)

    def get_dimension(self):
        return len(self.df.columns)

    def get_index_location(self, index_name):
        return PandasIndexManager().get_index_location(self.df, index_name)

    def get_array(self, column_name=None):
        if column_name is None:
            return np.array(self.df)
        else:
            return np.array(self.df[column_name])

    def convert_df_to_array(self):
        return PandasIndexManager().convert_df_to_array(self.df)

    def get_column_df(self, n_col=0):
        if isinstance(n_col, str):
            return self.df[n_col]
        else:
            return self.df[self.df.columns[n_col]]

    def get_rows(self, idx):
        return self.df.loc[idx, :]

    def get_row_by_idx_array(self, idx_array):
        return self.df.loc[list(map(tuple, np.array(idx_array))), :]

    def operation_on_idx(self, idx, func):
        self.df.loc[idx, :] = func(self.df.loc[idx, :])

    def operation(self, func):
        self.df = func(self.df)

    def operation_with_data_obj(self, obj, func, self_name_col=None, obj_name_col=None):
        if self_name_col is None:
            self_name_col = self.df.columns
        if obj_name_col is None:
            obj_name_col = self.df.columns

        self.df[self_name_col] = func(self.df[self_name_col], obj.df[obj_name_col])

    def print(self, short=True):
        if short:
            print(self.df.head())
        else:
            print(self.df)

    def get_unique_index_array(self, index_names=None):
        return PandasIndexManager().get_unique_index_array(df=self.df, index_names=index_names)

    def get_index_array(self, index_names=None):
        return PandasIndexManager().get_index_array(df=self.df, index_names=index_names)

    def get_index_dict(self, index_names):
        return PandasIndexManager().get_index_dict(df=self.df, index_names=index_names)

    def get_array_of_all_idx1_by_idx2(self, idx1_name, idx2_name, idx2_value):

        idx_array = self.get_unique_index_array()

        idx1_loc = self.get_index_location(idx1_name)
        idx2_loc = self.get_index_location(idx2_name)

        mask = np.where(idx_array[:, idx2_loc] == idx2_value)[0]
        res = set(idx_array[mask, idx1_loc])
        res = sorted(res)
        return np.array(res)

    def add_df_as_rows(self, df, replace=True):

        df.columns = self.df.columns
        self.df = PandasIndexManager.concat_dfs(self.df, df)

        if replace:
            idx_to_keep = ~self.df.index.duplicated(keep='last')
            self.df = self.df[idx_to_keep]

    def mean_over(self, index_name):
        idx_loc = self.get_index_location(index_name)
        index_names = self.df.index.names[:idx_loc+1]
        return self.df.mean(level=index_names)

    @staticmethod
    def __time_delta4each_group(df):
        df.iloc[:-1, :] = np.array(df.iloc[1:, :]) - np.array(df.iloc[:-1, :])
        df.iloc[-1, -1] = np.nan
        return df

    def compute_time_delta(self, index_name=id_frame_name):
        index_names = list(self.df.index.names)
        index_names.remove(index_name)
        return self.df.groupby(index_names).apply(self.__time_delta4each_group)

    def hist1d(self, column_name=None, bins='fd'):
        if column_name is None:
            if self.get_dimension() == 1:
                column_name = self.df.columns[0]
            else:
                raise IndexError('Data not 1d, precise on which column apply hist1d')

        y, x = np.histogram(self.df[column_name].dropna(), bins)
        x = (x[1:] + x[:-1]) / 2., 2
        h = np.array(list(zip(x, y)))

        df = PandasIndexManager().convert_array_to_df(array=h, index_names=column_name, column_names='Occurrences')
        return df.astype(int)

    def hist1d_evolution(self, column_name, index_name, start_index_intervals, end_index_intervals, bins, normed=False):
        if column_name is None:
            if len(self.df.columns) == 1:
                column_name = self.df.columns[0]
            else:
                raise IndexError('Data not 1d, precise on which column apply hist1d')

        if index_name is None:
            index_name = self.get_column_names()[-1]

        h = np.zeros((len(bins)-1, len(start_index_intervals)+1))
        h[:, 0] = (bins[1:] + bins[:-1]) / 2.

        for i in range(len(start_index_intervals)):
            index0 = start_index_intervals[i]
            index1 = end_index_intervals[i]

            index_location = (self.df[column_name].index.get_level_values(index_name) > index0)\
                & (self.df[column_name].index.get_level_values(index_name) < index1)

            df = self.df[column_name].loc[index_location]
            y, x = np.histogram(df.dropna(), bins, normed=normed)
            h[:, i+1] = y

        column_names = [
            str([start_index_intervals[i]/100., end_index_intervals[i]/100.])
            for i in range(len(start_index_intervals))]
        df = PandasIndexManager().convert_array_to_df(array=h, index_names='bins', column_names=column_names)
        if normed:
            return df
        else:
            return df.astype(int)

    def variance_evolution(self, column_name, index_name, start_index_intervals, end_index_intervals):
        if column_name is None:
            if len(self.df.columns) == 1:
                column_name = self.df.columns[0]
            else:
                raise IndexError('Data not 1d, precise on which column apply hist1d')

        if index_name is None:
            index_name = self.get_column_names()[-1]

        x = (end_index_intervals+start_index_intervals)/2.
        y = np.zeros(len(start_index_intervals))

        for i in range(len(start_index_intervals)):
            index0 = start_index_intervals[i]
            index1 = end_index_intervals[i]

            index_location = (self.df[column_name].index.get_level_values(index_name) > index0)\
                & (self.df[column_name].index.get_level_values(index_name) < index1)

            df = self.df[column_name].loc[index_location]
            y[i] = np.nanvar(df)

        df = pd.DataFrame(y, index=x)

        return df

    # def mean_over(self, level_df, mean_level=None, new_level_as=None):
    #     df = self.df.copy()
    #     filter_idx = 'new_idx'
    #     PandasIndexManager().add_index_level(
    #         df, filter_idx, np.array(level_df)
    #     )
    #     if mean_level is None:
    #         df = df.mean(level=[filter_idx])
    #     elif mean_level == 'exp':
    #         df = df.mean(level=[id_exp_name, filter_idx])
    #     elif mean_level == 'ant':
    #         df = df.mean(level=[id_exp_name, id_ant_name, filter_idx])
    #     else:
    #         raise ValueError(mean_level + ' not understood')
    #     if new_level_as is None:
    #         PandasIndexManager().remove_index_level(df, filter_idx)
    #     else:
    #         PandasIndexManager().rename_index_level(df, filter_idx, new_level_as)
    #     return df
