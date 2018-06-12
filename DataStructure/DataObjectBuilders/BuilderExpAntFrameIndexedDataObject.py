import pandas as pd

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
        return self.pandas_index_manager.get_dict_id_exp_ant_frame(self.df)

    def operation_on_id_exp(self, id_exp, fct):
        self.df.loc[id_exp, :, :] = fct(self.df.loc[id_exp, :, :])
