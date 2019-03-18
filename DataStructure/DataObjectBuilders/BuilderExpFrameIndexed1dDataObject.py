from DataStructure.DataObjectBuilders.Builder1dDataObject import Builder1dDataObject
from DataStructure.DataObjectBuilders.BuilderExpFrameIndexedDataObject import BuilderExpFrameIndexedDataObject
from DataStructure.VariableNames import id_exp_name, id_frame_name
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class BuilderExpFrameIndexed1dDataObject(Builder1dDataObject, BuilderExpFrameIndexedDataObject):
    def __init__(self, df):
        if df.index.names != [id_exp_name, id_frame_name]:
            raise IndexError('Index names are not (exp, frame)')
        else:
            Builder1dDataObject.__init__(self, df)
            BuilderExpFrameIndexedDataObject.__init__(self, df)

        self.pandas_index_manager = PandasIndexManager()
