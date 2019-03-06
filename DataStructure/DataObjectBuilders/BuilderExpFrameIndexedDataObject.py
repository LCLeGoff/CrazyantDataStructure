import pandas as pd
import numpy as np

from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class BuilderExpFrameIndexedDataObject:
    def __init__(self, df):
        if df.index.names != ['id_exp', 'frame']:
            raise IndexError('Index names are not (id_exp, frame)')
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
            return self.df.loc[pd.IndexSlice[id_exp, frame0:frame1], :]

    def operation_on_id_exp(self, id_exp, fct):
        self.df.loc[id_exp, :] = np.array(fct(self.df.loc[id_exp, :]))

    @staticmethod
    def __time_delta4each_group(df: pd.DataFrame):
        df.iloc[:-1, :] = np.array(df.iloc[1:, :]) - np.array(df.iloc[:-1, :])
        df.iloc[-1, -1] = np.nan
        return df

    def compute_time_delta(self):
        return self.df.groupby(['id_exp']).apply(self.__time_delta4each_group)
