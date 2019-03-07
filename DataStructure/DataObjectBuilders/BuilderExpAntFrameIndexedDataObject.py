import pandas as pd
import numpy as np

from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class BuilderExpAntFrameIndexedDataObject:
    def __init__(self, df):
        if df.index.names != ['id_exp', 'id_ant', 'frame']:
            raise IndexError('Index names are not (id_exp, id_ant, frame)')
        else:
            self.df = df
        self.pandas_index_manager = PandasIndexManager()

    def get_row_of_id_exp(self, id_exp):
        return self.df.loc[pd.IndexSlice[id_exp, :, :], :]

    def get_row_of_id_exp_ant(self, id_exp, id_ant):
        return self.df.loc[pd.IndexSlice[id_exp, id_ant, :], :]

    def get_row_of_id_exp_ant_frame(self, id_exp, id_ant, frame):
        return self.df.loc[pd.IndexSlice[id_exp, id_ant, frame], :]

    def get_row_of_id_exp_ant_in_frame_interval(self, id_exp, id_ant, frame0=None, frame1=None):
        if frame0 is None:
            frame0 = 0
        if frame1 is None:
            return self.df.loc[pd.IndexSlice[id_exp, id_ant, frame0:], :]
        else:
            return self.df.loc[pd.IndexSlice[id_exp, id_ant, frame0:frame1], :]

    def get_index_dict_of_id_exp_ant_frame(self):
        return self.pandas_index_manager.get_index_dict(df=self.df, index_names=['id_exp', 'id_ant', 'frame'])

    def operation_on_id_exp(self, id_exp, fct):
        self.df.loc[id_exp, :, :] = fct(self.df.loc[id_exp, :, :])

    @staticmethod
    def __time_delta4each_group(df: pd.DataFrame):
        df.iloc[:-1, :] = np.array(df.iloc[1:, :]) - np.array(df.iloc[:-1, :])
        df.iloc[-1, -1] = np.nan
        return df

    def compute_time_delta(self):
        return self.df.groupby(['id_exp', 'id_ant']).apply(self.__time_delta4each_group)
