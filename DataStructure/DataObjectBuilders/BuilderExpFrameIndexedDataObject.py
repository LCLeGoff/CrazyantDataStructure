import pandas as pd
import numpy as np

from DataStructure.VariableNames import id_exp_name, id_frame_name
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class BuilderExpFrameIndexedDataObject:
    def __init__(self, df):
        if df.index.names != [id_exp_name, id_frame_name]:
            raise IndexError('Index names are not (exp, frame)')
        else:
            self.df = df
        self.pandas_index_manager = PandasIndexManager()

    def get_row_of_id_exp(self, id_exp):
        return self.df.loc[pd.IndexSlice[id_exp, :], :]

    def get_row_of_id_exp_frame(self, id_exp, frame):
        return self.df.loc[pd.IndexSlice[id_exp, frame], :]

    def get_row_of_id_exp_in_frame_interval(self, id_exp, frame0=None, frame1=None):
        if frame0 is None:
            frame0 = 0
        if frame1 is None:
            return self.df.loc[pd.IndexSlice[id_exp, frame0:], :]
        else:
            return self.df.loc[pd.IndexSlice[id_exp, frame0:frame1-1], :]

    def operation_on_id_exp(self, id_exp, func):
        self.df.loc[id_exp, :] = np.array(func(self.df.loc[id_exp, :]))

    @staticmethod
    def __time_delta4each_group(df: pd.DataFrame):
        df.iloc[:-1, :] = np.array(df.iloc[1:, :]) - np.array(df.iloc[:-1, :])
        df.iloc[-1, -1] = np.nan
        return df

    def compute_time_delta(self):
        return self.df.groupby([id_exp_name]).apply(self.__time_delta4each_group)

    def hist1d_time_evolution(self, column_name, start_frame_intervals, end_frame_intervals, bins, normed=False):
        if column_name is None:
            if len(self.df.columns) == 1:
                column_name = self.df.columns[0]
            else:
                raise IndexError('Data not 1d, precise on which column apply hist1d')

        start_frame_intervals = np.array(start_frame_intervals, dtype=int)
        end_frame_intervals = np.array(end_frame_intervals, dtype=int)

        h = np.zeros((len(bins)-1, len(start_frame_intervals)+1))
        h[:, 0] = (bins[1:] + bins[:-1]) / 2.

        for i in range(len(start_frame_intervals)):
            frame0 = start_frame_intervals[i]
            frame1 = end_frame_intervals[i]

            df = self.df[column_name].loc[:, frame0:frame1]
            y, x = np.histogram(df.dropna(), bins, normed=normed)
            h[:, i+1] = y

        column_names = [
            str([start_frame_intervals[i]/100, end_frame_intervals[i]/100])
            for i in range(len(start_frame_intervals))]
        df = PandasIndexManager().convert_array_to_df(
                array=h, index_names='bins', column_names=column_names)
        if normed:
            return df
        else:
            return df.astype(int)

    def _apply_function_on_evolution(self, column_name, start_frame_intervals, end_frame_intervals, fct):
        if column_name is None:
            if len(self.df.columns) == 1:
                column_name = self.df.columns[0]
            else:
                raise IndexError('Data not 1d, precise on which column apply hist1d')
        start_frame_intervals = np.array(start_frame_intervals, dtype=int)
        end_frame_intervals = np.array(end_frame_intervals, dtype=int)
        x = (end_frame_intervals + start_frame_intervals) / 2. / 100.
        y = np.zeros(len(start_frame_intervals))
        for i in range(len(start_frame_intervals)):
            frame0 = start_frame_intervals[i]
            frame1 = end_frame_intervals[i]

            df = self.df[column_name].loc[:, frame0:frame1]
            y[i] = fct(df)
        df = pd.DataFrame(y, index=x)
        return df

    def sum_evolution(self, column_name, start_frame_intervals, end_frame_intervals):
        fct = np.nansum
        df = self._apply_function_on_evolution(column_name, start_frame_intervals, end_frame_intervals, fct)

        return df

    def mean_evolution(self, column_name, start_frame_intervals, end_frame_intervals):
        fct = np.nanmean
        df = self._apply_function_on_evolution(column_name, start_frame_intervals, end_frame_intervals, fct)

        return df

    def variance_evolution(self, column_name, start_frame_intervals, end_frame_intervals):
        fct = np.nanvar
        df = self._apply_function_on_evolution(column_name, start_frame_intervals, end_frame_intervals, fct)

        return df
