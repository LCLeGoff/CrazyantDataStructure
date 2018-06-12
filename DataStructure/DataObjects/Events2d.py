import pandas as pd

from DataStructure.DataObjectBuilders.BuilderExpAntFrameIndexed2dDataObject import BuilderExpAntFrameIndexed2dDataObject
from DataStructure.DataObjects.Definitions import DefinitionBuilder
from Tools.Plotter.Plotter2d import Plotter2d


class Events2d(BuilderExpAntFrameIndexed2dDataObject):
    def __init__(self, df, definition):
        BuilderExpAntFrameIndexed2dDataObject.__init__(self, df)
        DefinitionBuilder.build_from_definition(self, definition)
        self.plotter = Plotter2d(self)

    def copy(self, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
        return Events2dBuilder.build_from_df(
            self.df.copy(), name=name, xname=xname, yname=yname,
            category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description)


class Events2dBuilder:
    def __init__(self):
        pass

    @staticmethod
    def build_from_1d(
            event1, event2, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
        df = pd.DataFrame(index=event1.df.index)
        df[xname] = event1.df
        df[yname] = event2.df
        definition = DefinitionBuilder().build2d(
            name=name, category=category, object_type='Events2d',
            label=label, xlabel=xlabel, ylabel=ylabel, description=description
        )
        return Events2d(df, definition)

    @staticmethod
    def build_from_df(df, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
        df.columns = [xname, yname]
        definition = DefinitionBuilder().build2d(
            name=name, category=category, object_type='Events2d',
            label=label, xlabel=xlabel, ylabel=ylabel, description=description
        )
        return Events2d(df.sort_index(), definition)