import pandas as pd

import numpy as np

from DataStructure.DataObjectBuilders.Builder2dDataObject import Builder2dDataObject
from DataStructure.DataObjectBuilders.BuilderExpAntFrameIndexedDataObject import BuilderExpAntFrameIndexedDataObject
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.MiscellaneousTools.Fits import linear_fit, exp_fit, power_fit
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class BuilderExpAntFrameIndexed2dDataObject(Builder2dDataObject, BuilderExpAntFrameIndexedDataObject):
    def __init__(self, df):
        if df.index.names != [id_exp_name, id_ant_name, id_frame_name]:
            raise IndexError('Index names are not (id_exp, id_ant, frame)')
        else:
            Builder2dDataObject.__init__(self, df)
            BuilderExpAntFrameIndexedDataObject.__init__(self, df)

        self.pandas_index_manager = PandasIndexManager()

    def get_xy_values_of_id_exp_ant(self, id_exp, id_ant):
        df = self.get_row_of_id_exp_ant(id_exp, id_ant)
        return df[self.df.columns[0]],  df[self.df.columns[1]]

    def get_xy_values_of_id_exp(self, id_exp):
        df = self.get_row_of_id_exp(id_exp)
        return df[self.df.columns[0]],  df[self.df.columns[1]]

    def fit(self,
            level=None, typ='exp', filter_df=None,
            window=None, sqrt_x=False, sqrt_y=False, normed=False, list_id_exp=None):

        if list_id_exp is None:
            list_id_exp = self.get_index_array_of_id_exp()

        df_to_fit = self.df
        filter_idx_name = 'filter'

        if filter_df is not None:
            self.pandas_index_manager.add_index_level(
                df_to_fit, filter_idx_name, np.array(filter_df)
            )

        res = []
        idx_names = []

        if level is None:
            if filter_df is None:
                x = np.array(self.get_x_values())
                y = np.array(self.get_y_values())
                a, b = self._compute_result_fit(x, y, typ, window, sqrt_x, sqrt_y, normed)
                return [a, b]
            else:
                idx_names = [filter_idx_name]
                xy_array = self.convert_df_to_array()
                filter_value_set = set(xy_array[:, -3])
                for filter_val in filter_value_set:
                    x, y = self._get_xy_from_filter_val(filter_val, xy_array)
                    a, b = self._compute_result_fit(x, y, typ, window, sqrt_x, sqrt_y, normed)
                    res.append((filter_val, a, b))
        elif level == 'exp':
            if filter_df is None:
                idx_names = [id_exp_name]
                for id_exp in list_id_exp:
                    x, y = self.get_xy_values_of_id_exp(id_exp)
                    a, b = self._compute_result_fit(x, y, typ, window, sqrt_x, sqrt_y, normed)
                    res.append((id_exp, a, b))
            else:
                for id_exp in list_id_exp:
                    idx_names = [id_exp_name, filter_idx_name]
                    df = self.get_row_of_id_exp(id_exp)
                    xy_array = self.pandas_index_manager.convert_df_to_array(df)
                    filter_value_set = set(xy_array[:, -3])
                    for filter_val in filter_value_set:
                        x, y = self._get_xy_from_filter_val(filter_val, xy_array)
                        a, b = self._compute_result_fit(x, y, typ, window, sqrt_x, sqrt_y, normed)
                        res.append((id_exp, filter_val, a, b))
        elif level == 'ant':
            list_id_exp_ant = self.get_index_array_of_id_exp_ant()
            if filter_df is None:
                idx_names = [id_exp_name, id_ant_name]
                for id_exp, id_ant in list_id_exp_ant:
                    if id_exp in list_id_exp:
                        x, y = self.get_xy_values_of_id_exp(id_exp)
                        a, b = self._compute_result_fit(x, y, typ, window, sqrt_x, sqrt_y, normed)
                        res.append((id_exp, id_ant, a, b))
            else:
                idx_names = [id_exp_name, id_ant_name, filter_idx_name]
                for id_exp, id_ant in list_id_exp_ant:
                    if id_exp in list_id_exp:
                        df = self.get_row_of_id_exp_ant(id_exp, id_ant)
                        xy_array = self.pandas_index_manager.convert_df_to_array(df)
                        filter_value_set = set(xy_array[:, -3])
                        for filter_val in filter_value_set:
                            x, y = self._get_xy_from_filter_val(filter_val, xy_array)
                            a, b = self._compute_result_fit(x, y, typ, window, sqrt_x, sqrt_y, normed)
                            res.append((id_exp, id_ant, filter_val, a, b))
        else:
            raise ValueError(level+' not understood')

        if level is None:
            return res
        else:
            df_fit = self._set_fit_df(res, idx_names)
            return df_fit

    @staticmethod
    def _set_fit_df(res, idx_names):
        df = pd.DataFrame(res, columns=idx_names + ['a', 'b'])
        df.set_index(idx_names, inplace=True)
        return df

    @staticmethod
    def _get_xy_from_filter_val(filter_val, xy_array):
        idx_where_filter_val = np.where(xy_array[:, -3] == filter_val)[0]
        temp_xy_array = xy_array[idx_where_filter_val, :]
        x = temp_xy_array[:, -2]
        y = temp_xy_array[:, -1]
        return x, y

    def _compute_result_fit(self, x, y, typ, window, sqrt_x, sqrt_y, normed):
        x_fit, y_fit = self._set_x_fit_and_y_fit(x, y, window, sqrt_x, sqrt_y, normed)
        a, b, x_fit, y_fit = self._compute_fit_a_and_b(x_fit, y_fit, typ)
        return a, b

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
    def _compute_fit_a_and_b(x_fit, y_fit, typ):
        # pca = PCA(n_components=2)
        # tab = np.c_[x_fit, y_fit]
        # print(tab)
        # pca.fit(tab)
        # axis = pca.components_.T
        # x_fit = axis[:, 0]
        # y_fit = axis[:, 1]
        mask = np.where(y_fit != 0)[0]
        if len(mask) != 0:
            if typ == 'linear':
                return linear_fit(x_fit[mask], y_fit[mask])
            elif typ == 'exp':
                return exp_fit(x_fit[mask], y_fit[mask])
            elif typ == 'power':
                return power_fit(x_fit[mask], y_fit[mask])
            else:
                raise TypeError(typ + ' is not a type of fit known')
        else:
            return np.nan, np.nan, [], []
