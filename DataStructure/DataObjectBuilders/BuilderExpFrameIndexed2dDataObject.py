from DataStructure.DataObjectBuilders.Builder2dDataObject import Builder2dDataObject
from DataStructure.DataObjectBuilders.BuilderExpFrameIndexedDataObject import BuilderExpFrameIndexedDataObject
from DataStructure.VariableNames import id_exp_name, id_frame_name
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class BuilderExpFrameIndexed2dDataObject(Builder2dDataObject, BuilderExpFrameIndexedDataObject):
    def __init__(self, df):
        if df.index.names != [id_exp_name, id_frame_name]:
            raise IndexError('Index names are not (exp, frame)')
        else:
            Builder2dDataObject.__init__(self, df)
            BuilderExpFrameIndexedDataObject.__init__(self, df)

        self.pandas_index_manager = PandasIndexManager()
