from DataStructure.DataObjectBuilders.Builder2dDataObject import Builder2dDataObject
from DataStructure.DataObjectBuilders.BuilderExpFrameIndexedDataObject import BuilderExpFrameIndexedDataObject
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class BuilderExpFrameIndexed2dDataObject(Builder2dDataObject, BuilderExpFrameIndexedDataObject):
    def __init__(self, df):
        if df.index.names != ['id_exp', 'frame']:
            raise IndexError('Index names are not (id_exp, frame)')
        else:
            Builder2dDataObject.__init__(self, df)
            BuilderExpFrameIndexedDataObject.__init__(self, df)

        self.pandas_index_manager = PandasIndexManager()
