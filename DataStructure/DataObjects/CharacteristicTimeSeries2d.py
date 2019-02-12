import pandas as pd

from DataStructure.DataObjectBuilders.BuilderExpFrameIndexed2dDataObject import BuilderExpFrameIndexed2dDataObject
from DataStructure.DataObjects.Definitions import DefinitionBuilder
from Tools.Plotter.Plotter2d import Plotter2d


class CharacteristicTimeSeries2d(BuilderExpFrameIndexed2dDataObject):
    def __init__(self, df, definition):
        BuilderExpFrameIndexed2dDataObject.__init__(self, df)
        DefinitionBuilder.add_definition_to_class(self, definition)
        self.plotter = Plotter2d(self)

    def rename(
            self, name, xname=None, yname=None, category=None, label=None, xlabel=None, ylabel=None, description=None):

        if xname is None:
            xname = self.xname
        if yname is None:
            yname = self.yname
        if category is None:
            category = self.category
        if label is None:
            label = self.label
        if xlabel is None:
            xlabel = self.xlabel
        if ylabel is None:
            ylabel = self.ylabel
        if description is None:
            description = self.description

        self.rename_df(xname, yname)

        definition = DefinitionBuilder().build2d(
            name=name, object_type='CharacteristicTimeSeries2d', category=category,
            label=label, xlabel=xlabel, ylabel=ylabel, description=description)
        DefinitionBuilder.add_definition_to_class(self, definition=definition)

    def copy(
            self, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
        return CharacteristicTimeSeries2dBuilder.build_from_df(
            df=self.df.copy(), name=name, xname=xname, yname=yname,
            category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description)


class CharacteristicTimeSeries2dBuilder:
    def __init__(self):
        pass

    @staticmethod
    def build_from_1d(ts1, ts2, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None,
                      description=None):
        df = pd.DataFrame(index=ts1.df.index)
        df[xname] = ts1.df
        df[yname] = ts2.df
        definition = DefinitionBuilder().build2d(
            name=name, category=category, object_type='CharacteristicTimeSeries2d',
            label=label, xlabel=xlabel, ylabel=ylabel, description=description
        )
        return CharacteristicTimeSeries2d(df, definition)

    @staticmethod
    def build_from_df(df, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
        df.columns = [xname, yname]
        definition = DefinitionBuilder().build2d(
            name=name, category=category, object_type='CharacteristicTimeSeries2d',
            label=label, xlabel=xlabel, ylabel=ylabel, description=description
        )
        return CharacteristicTimeSeries2d(df.sort_index(), definition)
