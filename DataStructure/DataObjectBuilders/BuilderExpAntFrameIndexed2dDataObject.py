from DataStructure.DataObjectBuilders.Builder2dDataObject import Builder2dDataObject
from DataStructure.DataObjectBuilders.BuilderExpAntFrameIndexedDataObject import BuilderExpAntFrameIndexedDataObject
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
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
