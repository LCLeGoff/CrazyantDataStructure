from DataStructure.DataObjectBuilders.Builder1dDataObject import Builder1dDataObject
from DataStructure.DataObjectBuilders.BuilderExpAntFrameIndexedDataObject import BuilderExpAntFrameIndexedDataObject
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class BuilderExpAntFrameIndexed1dDataObject(Builder1dDataObject, BuilderExpAntFrameIndexedDataObject):
    def __init__(self, df):
        if df.index.names != ['id_exp', 'id_ant', 'frame']:
            raise IndexError('Index names are not (id_exp, id_ant, frame)')
        else:
            Builder1dDataObject.__init__(self, df)
            BuilderExpAntFrameIndexedDataObject.__init__(self, df)

        self.pandas_index_manager = PandasIndexManager()
