from DataStructure.DataObjectBuilders.Builder1dDataObject import Builder1dDataObject
from DataStructure.DataObjectBuilders.BuilderExpAntFrameIndexedDataObject import BuilderExpAntFrameIndexedDataObject
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class BuilderExpAntFrameIndexed1dDataObject(Builder1dDataObject, BuilderExpAntFrameIndexedDataObject):
    def __init__(self, df):
        if df.index.names != [id_exp_name, id_ant_name, id_frame_name]:
            raise IndexError('Index names are not (exp, ant, frame)')
        else:
            Builder1dDataObject.__init__(self, df)
            BuilderExpAntFrameIndexedDataObject.__init__(self, df)

        self.pandas_index_manager = PandasIndexManager()
