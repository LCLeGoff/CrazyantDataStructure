import pandas as pd

from DataStructure.DataObjectBuilders.BuilderExpAntFrameIndexed2dDataObject import BuilderExpAntFrameIndexed2dDataObject
from DataStructure.DataObjects.Definitions import DefinitionBuilder
from Tools.Plotter.Plotter2d import Plotter2d


class TimeSeries2d(BuilderExpAntFrameIndexed2dDataObject):
    def __init__(self, df, definition):
        BuilderExpAntFrameIndexed2dDataObject.__init__(self, df)
        DefinitionBuilder.build_from_definition(self, definition)
        self.plotter = Plotter2d(self)


class TimeSeries2dBuilder:
    def __init__(self):
        pass

    @staticmethod
    def build_from_1d(ts1, ts2, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None,
                      description=None):
        df = pd.DataFrame(index=ts1.df.index)
        df[xname] = ts1.df
        df[yname] = ts2.df
        definition = DefinitionBuilder().build2d(
            name=name, category=category, object_type='TimeSeries2d',
            label=label, xlabel=xlabel, ylabel=ylabel, description=description
        )
        return TimeSeries2d(df, definition)

    @staticmethod
    def build_from_df(df, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
        df.columns = [xname, yname]
        definition = DefinitionBuilder().build2d(
            name=name, category=category, object_type='TimeSeries2d',
            label=label, xlabel=xlabel, ylabel=ylabel, description=description
        )
        return TimeSeries2d(df.sort_index(), definition)
