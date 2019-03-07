import numpy as np

from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class OneIndexedDataSetDecorator:
    def __init__(self, df):
        if df.index.names != ['id_exp']:
            raise IndexError('Index names are not id_exp')
        else:
            self.df = df

    def rename_df(self, names):
        if isinstance(names, str):
            names = [names]
        self.df.columns = names

    def get_index_location(self, index_name):
        return PandasIndexManager().get_index_location(self.df, index_name)

    def get_array(self):
        return np.array(self.df)

    def convert_df_to_array(self):
        return PandasIndexManager().convert_df_to_array(self.df)

    def get_column_values(self, n_col=0):
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

    def operation_with_data_obj(self, obj, fct, self_name_col=None, obj_name_col=None):
        if self_name_col is None:
            self_name_col = self.df.columns
        if obj_name_col is None:
            obj_name_col = self.df.columns

        self.df[self_name_col] = fct(self.df[self_name_col], obj.df[obj_name_col])

    def print(self, short=True):
        if short:
            print(self.df.head())
        else:
            print(self.df)

    def get_index_array(self, index_names=None):
        return PandasIndexManager().get_index_array(df=self.df, index_names=index_names)

    def get_index_dict(self, index_names):
        return PandasIndexManager().get_index_dict(df=self.df, index_names=index_names)

    def get_array_of_all_idx1_by_idx2(self, idx1_name, idx2_name, idx2_value):

        idx_array = self.get_index_array()

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

    # def mean_over(self, level_df, mean_level=None, new_level_as=None):
    #     df = self.df.copy()
    #     filter_idx = 'new_idx'
    #     PandasIndexManager().add_index_level(
    #         df, filter_idx, np.array(level_df)
    #     )
    #     if mean_level is None:
    #         df = df.mean(level=[filter_idx])
    #     elif mean_level == 'exp':
    #         df = df.mean(level=['id_exp', filter_idx])
    #     elif mean_level == 'ant':
    #         df = df.mean(level=['id_exp', 'id_ant', filter_idx])
    #     else:
    #         raise ValueError(mean_level + ' not understood')
    #     if new_level_as is None:
    #         PandasIndexManager().remove_index_level(df, filter_idx)
    #     else:
    #         PandasIndexManager().rename_index_level(df, filter_idx, new_level_as)
    #     return df
