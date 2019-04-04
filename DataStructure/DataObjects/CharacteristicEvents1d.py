from DataStructure.DataObjectBuilders.BuilderExpFrameIndexed1dDataObject import BuilderExpFrameIndexed1dDataObject
from DataStructure.DataObjects.Definitions import DefinitionBuilder
from Tools.Plotter.Plotter1d import Plotter1d


class CharacteristicEvents1d(BuilderExpFrameIndexed1dDataObject):
    def __init__(self, df, definition):
        BuilderExpFrameIndexed1dDataObject.__init__(self, df)
        DefinitionBuilder.add_definition_to_class(self, definition)
        self.plotter = Plotter1d(self)

    def rename(self, name, category=None, label=None, description=None):

        if category is None:
            category = self.category
        if label is None:
            label = self.label
        if description is None:
            description = self.description

        self.rename_df(name)
        definition = DefinitionBuilder().build1d(
            name=name, object_type='CharacteristicEvents1d', category=category, label=label, description=description)
        DefinitionBuilder.add_definition_to_class(self, definition=definition)

    def copy(self, name, category=None, label=None, description=None):
        return CharacteristicEvents1dBuilder.build(
            df=self.df.copy(), name=name, category=category, label=label, description=description)


class CharacteristicEvents1dBuilder:
    def __init__(self):
        pass

    @staticmethod
    def build(df, name, category=None, label=None, description=None):
        definition = DefinitionBuilder().build1d(
            name=name, category=category, object_type='CharacteristicEvents1d',
            label=label, description=description
        )
        df.columns = [name]
        return CharacteristicEvents1d(df.sort_index(), definition)
