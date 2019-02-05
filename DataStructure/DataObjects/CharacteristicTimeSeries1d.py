import numpy as np

from DataStructure.DataObjectBuilders.BuilderExpFrameIndexed1dDataObject import BuilderExpFrameIndexed1dDataObject
from DataStructure.DataObjects.Definitions import DefinitionBuilder
from DataStructure.DataObjects.Events1d import Events1dBuilder
from Tools.Plotter.Plotter1d import Plotter1d


class CharacteristicTimeSeries1d(BuilderExpFrameIndexed1dDataObject):
    def __init__(self, df, definition):
        # TODO: Fix this conception mistake
        df.columns = [definition.name]
        BuilderExpFrameIndexed1dDataObject.__init__(self, df)
        DefinitionBuilder.build_from_definition(self, definition)
        self.plotter = Plotter1d(self)

    def copy(self, name, category=None, label=None, description=None):
        return CharacteristicTimeSeries1dBuilder.build(
            df=self.df.copy(), name=name, category=category, label=label, description=description)

    def extract_event(self, name, category=None, label=None, description=None):
        ts_array = self.get_array()
        ts_array2 = ts_array[1:] - ts_array[:-1]
        mask = [0] + list(np.where(ts_array2 != 0)[0] + 1)
        event = self.df.iloc[mask]
        event.columns = [name]
        return Events1dBuilder().build(df=event, name=name, category=category, label=label, description=description)


class CharacteristicTimeSeries1dBuilder:
    def __init__(self):
        pass

    @staticmethod
    def build(df, name, category=None, label=None, description=None):
        definition = DefinitionBuilder().build1d(
            name=name, category=category, object_type='CharacteristicTimeSeries1d',
            label=label, description=description
        )
        df.columns = [name]
        return CharacteristicTimeSeries1d(df.sort_index(), definition)
