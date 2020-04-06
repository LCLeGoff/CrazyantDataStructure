import numpy as np
import pandas as pd

from DataStructure.VariableNames import id_frame_name
from Tools.MiscellaneousTools import Fits
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
            self_name_col = self.df.columns[0]
        if obj_name_col is None:
            obj_name_col = obj.df.columns[0]

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

    def hist1d(self, column_name=None, bins='fd', error=False):
        if column_name is None:
            if self.get_dimension() == 1:
                column_name = self.df.columns[0]
            else:
                raise IndexError('Data not 1d, precise on which column apply hist1d')

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

        df = pd.DataFrame(h, index=y, columns=x)
        return df.astype(int)

    def survival_curve(self, start, column_name=None):
        if column_name is None:
            if self.get_dimension() == 1:
                column_name = self.df.columns[0]
            else:
                raise IndexError('Data not 1d, precise on which column apply hist1d')

        vals = self.df[column_name].dropna().values
        x = np.sort(vals)
        if x[0] > start:
            x[0] = start
        y = 1-np.arange(len(vals)) / (len(vals) - 1)
        tab = np.array(list(zip(x, y)))

        df = PandasIndexManager().convert_array_to_df(array=tab, index_names=column_name, column_names='Survival')
        return df

    @staticmethod
    def get_column_name(column_name, df):
        if column_name is None:
            if len(df.columns) == 1:
                column_name = df.columns[0]
            else:
                raise IndexError('Data not 1d, precise on which column apply hist1d')
        return column_name

    def vs(self, df2, column_name=None, column_name2=None, n_bins=10, x_are_integers=False):
        column_name = self.get_column_name(column_name, self.df)
        column_name2 = self.get_column_name(column_name2, df2)

        x_min = float(self.df[column_name].min())
        x_max = float(self.df[column_name].max())

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

        val_dict = dict()
        for x in m_bins:
            val_dict[x] = []

        for val in df2[column_name2]:
            idx = int(np.floor(val/dx))
            val_dict[m_bins[idx]].append(val)

        lg = len(m_bins)
        res = np.zeros((lg, 3))
        for j, x in enumerate(m_bins):
            list_vals = val_dict[x]
            if len(list_vals) == 0:
                res[j, 0] = 0.
                res[j, 1] = 0.
                res[j, 2] = 0.
            else:
                res[j, 0] = np.mean(list_vals)
                res[j, 1] = - np.percentile(list_vals, 2.5) + res[j, 0]
                res[j, 2] = np.percentile(list_vals, 97.5) - res[j, 0]

        res_df = pd.DataFrame(res, index=m_bins, columns=[column_name, 'error1', 'error2'])
        return res_df

    def hist1d_evolution(
            self, column_name, index_name, start_frame_intervals, end_frame_intervals, bins, fps=100., normed=False):

        column_name = self.get_column_name(column_name, self.df)

        if index_name is None:
            index_name = self.df.index.names[-1]

        h = np.zeros((len(bins) - 1, len(start_frame_intervals) + 1))
        h[:, 0] = (bins[1:] + bins[:-1]) / 2.

        for i in range(len(start_frame_intervals)):
            index0 = start_frame_intervals[i]
            index1 = end_frame_intervals[i]

            index_location = (self.df[column_name].index.get_level_values(index_name) >= index0)\
                & (self.df[column_name].index.get_level_values(index_name) < index1)

            df = self.df[column_name].loc[index_location]
            y, x = np.histogram(df.dropna(), bins, normed=normed)
            h[:, i+1] = y

        column_names = [
            str([start_frame_intervals[i] / fps, end_frame_intervals[i] / fps])
            for i in range(len(start_frame_intervals))]
        df = PandasIndexManager().convert_array_to_df(array=h, index_names='bins', column_names=column_names)
        if normed:
            return df
        else:
            return df.astype(int)

    def variance_evolution(self, column_name, index_name, start_index_intervals, end_index_intervals):
        
        fct = np.nanvar
        df = self._apply_function_on_evolution(index_name, column_name, fct, start_index_intervals, end_index_intervals)

        return df
    
    def mean_evolution(self, column_name, index_name, start_index_intervals, end_index_intervals):
        
        fct = np.nanmean
        df = self._apply_function_on_evolution(index_name, column_name, fct, start_index_intervals, end_index_intervals)

        return df

    def sum_evolution(self, column_name, index_name, start_index_intervals, end_index_intervals):

        fct = np.nansum
        df = self._apply_function_on_evolution(index_name, column_name, fct, start_index_intervals, end_index_intervals)

        return df

    def _apply_function_on_evolution(self, index_name, column_name, fct, start_index_intervals, end_index_intervals):
        if column_name is None:
            if len(self.df.columns) == 1:
                column_name = self.df.columns[0]
            else:
                raise IndexError('Data not 1d, precise on which column apply the method')
        if index_name is None:
            index_name = self.get_column_names()[-1]
        x = (end_index_intervals + start_index_intervals) / 2.
        y = np.zeros(len(start_index_intervals))
        for i in range(len(start_index_intervals)):
            index0 = start_index_intervals[i]
            index1 = end_index_intervals[i]

            index_location = (self.df[column_name].index.get_level_values(index_name) >= index0) \
                & (self.df[column_name].index.get_level_values(index_name) < index1)

            df = self.df[column_name].loc[index_location]
            y[i] = fct(df)
        df = pd.DataFrame(y, index=x)
        return df

    def rolling_mean(self, window, index_names):

        window = int(np.floor(window / 2) * 2 + 1)
        df_res = self.df.groupby(index_names).rolling(window, center=True).mean()
        for _ in range(len(index_names)):
            df_res = df_res.reset_index(0, drop=True)

        return df_res.round(6)

    def rolling_mean_angle(self, window, index_names):
        #  Bug with complex numbers and rolling. So need another algo than in rolling_mean
        window = int(np.floor(window / 2) * 2 + 1)

        df_nan = self.df.isna()
        df2 = self.df.copy()
        df2['cos'] = np.cos(self.df.values)
        df2['sin'] = np.sin(self.df.values)
        df2 = df2.drop(columns=self.df.columns[0])

        df2 = df2.groupby(index_names).rolling(window, center=True).mean()
        df2 = df2.reset_index(0, drop=True).reset_index(0, drop=True)

        df_res = self.df.copy()
        df_res[:] = np.c_[np.arctan2(df2['sin'], df2['cos'])]
        df_res[df_nan] = np.nan

        return df_res.round(6)

    def fit(self, typ='exp', window=None, sqrt_x=False, sqrt_y=False, normed=False, column=None, cst=None):
        x = np.array(list(self.df.index))
        if column is None:
            y = self.df.values.ravel()
        else:
            y = self.df[column].values.ravel()

        return self._compute_result_fit(x, y, typ, window, sqrt_x, sqrt_y, normed, cst=cst)

    def _compute_result_fit(self, x, y, typ, window, sqrt_x, sqrt_y, normed, cst=None):
        x_fit, y_fit = self._set_x_fit_and_y_fit(x, y, window, sqrt_x, sqrt_y, normed)
        return self._compute_fit_a_and_b(x_fit, y_fit, typ, cst=cst)

    @staticmethod
    def _set_x_fit_and_y_fit(x, y, window, sqrt_x, sqrt_y, normed):
        if sqrt_x:
            x_fit = np.sqrt(x)
        else:
            x_fit = np.array(x)

        if normed is True:
            s = float(sum(y))
            y_fit = np.array(y) / s
        elif normed == 'density':
            y_fit = np.array(y) / float(sum(y)) / float(x[1] - x[0])
        else:
            y_fit = np.array(y)
        if sqrt_y is True:
            y_fit = np.sqrt(y_fit)
        else:
            y_fit = np.array(y_fit)

        if window is not None:
            x_fit = x_fit[window[0]: window[1]]
            y_fit = y_fit[window[0]: window[1]]
        return x_fit, y_fit

    @staticmethod
    def _compute_fit_a_and_b(x_fit, y_fit, typ, cst=None):
        mask = np.where(y_fit != 0)[0]
        if len(mask) != 0:
            if typ == 'linear':
                return Fits.linear_fit(x_fit[mask], y_fit[mask])
            elif typ == 'exp':
                return Fits.exp_fit(x_fit[mask], y_fit[mask], cst=cst)
            elif typ == 'power':
                return Fits.power_fit(x_fit[mask], y_fit[mask])
            elif typ == 'cst center gauss':
                return Fits.centered_gauss_cst_fit(x_fit[mask], y_fit[mask], p0=cst)
            else:
                raise TypeError(typ + ' is not a type of fit known')
        else:
            return np.nan, np.nan, [], []

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
