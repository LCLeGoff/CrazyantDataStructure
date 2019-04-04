import pandas as pd

from DataStructure.DataObjectBuilders.BuilderExpFrameIndexed2dDataObject import BuilderExpFrameIndexed2dDataObject
from DataStructure.DataObjects.Definitions import DefinitionBuilder
from Tools.Plotter.SetupPlotter import SetupPlotter


class CharacteristicEvents2d(BuilderExpFrameIndexed2dDataObject):
    def __init__(self, df, definition):
        BuilderExpFrameIndexed2dDataObject.__init__(self, df)
        DefinitionBuilder.add_definition_to_class(self, definition)
        self.plotter = SetupPlotter(self)

    def rename(
            self, name, xname=None, yname=None, category=None, label=None, xlabel=None, ylabel=None, description=None):
        if xname is None:
            xname = self.df.columns[0]
        if yname is None:
            yname = self.df.columns[1]
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
            name=name, object_type='CharacteristicEvents2d', category=category,
            label=label, xlabel=xlabel, ylabel=ylabel, description=description)
        DefinitionBuilder.add_definition_to_class(self, definition=definition)

    def copy(self, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
        return CharacteristicEvents2dBuilder.build_from_df(
            self.df.copy(), name=name, xname=xname, yname=yname,
            category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description)


class CharacteristicEvents2dBuilder:
    def __init__(self):
        pass

    @staticmethod
    def build_from_1d(
            event1, event2, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
        df = pd.DataFrame(index=event1.df.index)
        df[xname] = event1.df
        df[yname] = event2.df
        definition = DefinitionBuilder().build2d(
            name=name, category=category, object_type='CharacteristicEvents2d',
            label=label, xlabel=xlabel, ylabel=ylabel, description=description
        )
        return CharacteristicEvents2d(df, definition)

    @staticmethod
    def build_from_df(df, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
        df.columns = [xname, yname]
        definition = DefinitionBuilder().build2d(
            name=name, category=category, object_type='CharacteristicEvents2d',
            label=label, xlabel=xlabel, ylabel=ylabel, description=description
        )
        return CharacteristicEvents2d(df.sort_index(), definition)
