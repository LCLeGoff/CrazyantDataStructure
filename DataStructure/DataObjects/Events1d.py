from DataStructure.DataObjectBuilders.BuilderExpAntFrameIndexed1dDataObject import BuilderExpAntFrameIndexed1dDataObject
from DataStructure.DataObjects.Definitions import DefinitionBuilder
from Tools.Plotter.Plotter1d import Plotter1d


class Events1d(BuilderExpAntFrameIndexed1dDataObject):
    def __init__(self, df, definition):
        BuilderExpAntFrameIndexed1dDataObject.__init__(self, df)
        DefinitionBuilder.build_from_definition(self, definition)
        self.plotter = Plotter1d(self)

    def copy(self, name, category=None, label=None, description=None):
        return Events1dBuilder.build(
            df=self.df.copy(), name=name, category=category, label=label, description=description)


class Events1dBuilder:
    def __init__(self):
        pass

    @staticmethod
    def build(df, name, category=None, label=None, description=None):
        definition = DefinitionBuilder().build1d(
            name=name, category=category, object_type='Events1d',
            label=label, description=description
        )
        df.columns = [name]
        return Events1d(df.sort_index(), definition)