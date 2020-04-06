import numpy as np
import pandas as pd

from DataStructure.VariableNames import id_ant_name, id_frame_name, id_exp_name
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class BuilderDataObject:
    def __init__(self, df):
        self.df = df
        self.pandas_index_manager = PandasIndexManager()

    def operation(self, func):
        self.df = func(self.df)

    def operation_with_data_obj(self, obj, func, self_name_col=None, obj_name_col=None):
        if self_name_col is None:
            self_name_col = self.df.columns[0]
        if obj_name_col is None:
            obj_name_col = obj.name_col

        self.df[self_name_col] = func(self.df[self_name_col], obj.df[obj_name_col])

    def print(self, short=True):
        if short:
            print(self.df.head())
        else:
            print(self.df)

    def convert_df_to_array(self):
        return self.pandas_index_manager.convert_df_to_array(self.df)

    def get_dimension(self):
        return len(self.df.columns)

    def get_nbr_index(self):
        return len(self.df.index.names)

    def get_row(self, idx):
        return self.df.loc[idx, :]

    def get_row_of_idx_array(self, idx_array):
        return self.df.loc[list(map(tuple, np.array(idx_array))), :]

    def get_index_array(self, index_names=None):
        return PandasIndexManager().get_index_array(df=self.df, index_names=index_names)

    def get_index_array_of_id_exp(self):
        return PandasIndexManager().get_unique_index_array(self.df, id_exp_name)

    def get_index_array_of_id_exp_ant(self):
        return PandasIndexManager().get_unique_index_array(self.df, [id_exp_name, id_ant_name])

    def get_index_array_of_id_exp_frame(self):
        return PandasIndexManager().get_unique_index_array(self.df, [id_exp_name, id_frame_name])

    def get_index_array_of_id_exp_ant_frame(self):
        return PandasIndexManager().get_unique_index_array(self.df, [id_exp_name, id_ant_name, id_frame_name])

    def get_index_dict_of_id_exp_ant(self):
        return PandasIndexManager().get_index_dict(self.df, [id_exp_name, id_ant_name])

    def get_index_dict_of_id_exp_frame(self):
        return PandasIndexManager().get_index_dict(self.df, [id_exp_name, id_frame_name])

    def get_index_dict_of_id_exp_ant_frame(self):
        return PandasIndexManager().get_index_dict(self.df, [id_exp_name, id_ant_name, id_frame_name])

    def get_array_of_all_ants_of_exp(self, id_exp):
        id_exp_ant_frame_array = self.get_index_array_of_id_exp_ant_frame()
        idx_where_id_exp = np.where(id_exp_ant_frame_array[:, 0] == id_exp)
        res = set(id_exp_ant_frame_array[idx_where_id_exp, 1])
        res = sorted(res)
        return np.array(res)

    def get_array_of_all_frames_of_exp(self, id_exp):
        if id_ant_name not in self.df.columns:
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

    def mean_over_exp_and_ants(self):
        return self.df.mean(level=[id_exp_name, id_ant_name])

    def mean_over_experiments(self):
        return self.df.mean(level=id_exp_name)

    def sum_over_exp_and_ants(self):
        return self.df.sum(level=[id_exp_name, id_ant_name])

    def sum_over_experiments(self):
        return self.df.sum(level=id_exp_name)

    def sum_over_ants(self):
        return self.df.sum(level=id_ant_name)

    def mean_over_exp_and_frames(self):
        return self.df.mean(level=[id_exp_name, id_frame_name])

    def sum_over_exp_and_frames(self):
        return self.df.sum(level=[id_exp_name, id_frame_name])

    def mean_over(self, level_df, mean_level=None, new_level_as=None):
        df = self.df.copy()
        filter_idx = 'new_idx'
        self.pandas_index_manager.add_index_level(
            df, filter_idx, np.array(level_df)
        )
        if mean_level is None:
            df = df.mean(level=[filter_idx])
        elif mean_level == 'exp':
            df = df.mean(level=[id_exp_name, filter_idx])
        elif mean_level == 'ant':
            df = df.mean(level=[id_exp_name, id_ant_name, filter_idx])
        else:
            raise ValueError(mean_level + ' not understood')
        if new_level_as is None:
            self.pandas_index_manager.remove_index_level(df, filter_idx)
        else:
            self.pandas_index_manager.rename_index_level(df, filter_idx, new_level_as)
        return df

    def hist1d(self, column_name=None, bins='fd', error=False):
        column_name = self.get_column_name(column_name, self.df)

        vals = self.df[column_name].dropna()
        y, x = np.histogram(vals, bins)
        x = (x[1:] + x[:-1]) / 2.

        if error is True:
            n = len(vals)
            h = np.zeros((len(x), 4))
            h[:, 0] = x
            h[:, 1] = y/n
            std = np.sqrt(h[:, 1]*(1-h[:, 1])/n)
            h[:, 2] = 1.95*std
            h[:, 3] = 1.95*std

            df = PandasIndexManager().convert_array_to_df(
                array=h, index_names=column_name, column_names=['PDF', 'err1', 'err2'])
            return df
        else:
            h = np.array(list(zip(x, y)))
            df = PandasIndexManager().convert_array_to_df(array=h, index_names=column_name, column_names='Occurrences')
            return df.astype(int)

    def hist2d(self, df2, column_name=None, column_name2=None, bins=10):
        column_name = self.get_column_name(column_name, self.df)
        column_name2 = self.get_column_name(column_name2, df2)

        df_nonan = self.df[column_name].dropna()
        df2_nonan = df2[column_name2].dropna()
        idx = df_nonan.index.intersection(df2_nonan.index)

        h, x, y = np.histogram2d(df2_nonan.reindex(idx), df_nonan.reindex(idx), bins)

        x = (x[1:]+x[:-1])/2.
        y = (y[1:]+y[:-1])/2.
        df = pd.DataFrame(h, index=y, columns=x)
        return df.astype(int)

    def survival_curve(self, start, column_name=None):
        column_name = self.get_column_name(column_name, self.df)

        vals = self.df[column_name].dropna().values
        x = np.sort(vals)
        if x[0] > start:
            x[0] = start
        y = 1-np.arange(len(vals)) / (len(vals) - 1)
        tab = np.array(list(zip(x, y)))

        df = PandasIndexManager().convert_array_to_df(array=tab, index_names=column_name, column_names='Survival')
        return df

    @staticmethod
    def get_column_name(column_name2, df2):
        if column_name2 is None:
            if len(df2.columns) == 1:
                column_name2 = df2.columns[0]
            else:
                raise IndexError('Data not 1d, precise on which column apply hist1d')
        return column_name2

    def vs(self, df2, column_name=None, column_name2=None, n_bins=10, x_are_integers=False):
        column_name = self.get_column_name(column_name, self.df)
        column_name2 = self.get_column_name(column_name2, df2)

        df3 = df2.reindex(self.df.index)

        x_min = np.floor(float(np.nanmin(df3[column_name2])))
        x_max = np.ceil(float(np.nanmax(df3[column_name2])))

        if isinstance(n_bins, int):
            dx = (x_max - x_min+1) / n_bins
            bins = np.arange(x_min, x_max+dx, dx)
        else:
            bins = np.array(n_bins)
            dx = bins[1]-bins[0]

        if x_are_integers:
            m_bins = bins[:-1]
        else:
            m_bins = (bins[1:] + bins[:-1]) / 2.
        lg = len(m_bins)

        val_dict = dict()
        for x in m_bins:
            val_dict[x] = []

        for i_val, val in enumerate(df3[column_name2]):
            if not np.isnan(val):
                idx = int(np.floor(val / dx))
                idx = min(lg-1, idx)

                val_dict[m_bins[idx]].append(self.df[column_name].iloc[i_val])

        res = np.zeros((lg, 3))
        for j, x in enumerate(m_bins):
            list_vals = val_dict[x]
            if len(list_vals) == 0:
                res[j, 0] = 0.
                res[j, 1] = 0.
                res[j, 2] = 0.
            else:
                res[j, 0] = np.nanmean(list_vals)
                res[j, 1] = - np.nanpercentile(list_vals, 2.5) + res[j, 0]
                res[j, 2] = np.nanpercentile(list_vals, 97.5) - res[j, 0]

        res_df = pd.DataFrame(res, index=m_bins, columns=[column_name, 'error1', 'error2'])
        return res_df
