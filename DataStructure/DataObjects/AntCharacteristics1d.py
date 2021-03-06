from DataStructure.DataObjectBuilders.Builder1dDataObject import Builder1dDataObject
from DataStructure.DataObjectBuilders.BuilderExpAntIndexedDataObject import BuilderExpAntIndexedDataObject
from DataStructure.DataObjects.Definitions import DefinitionBuilder


class AntCharacteristics1d(Builder1dDataObject, BuilderExpAntIndexedDataObject):
    def __init__(self, df, definition):
        Builder1dDataObject.__init__(self, df)
        BuilderExpAntIndexedDataObject.__init__(self, df)
        DefinitionBuilder.add_definition_to_class(self, definition)

    def rename(self, name, category=None, label=None, description=None):

        if category is None:
            category = self.category
        if label is None:
            label = self.label
        if description is None:
            description = self.description

        self.rename_df(name)
        definition = DefinitionBuilder().build1d(
            name=name, object_type='AntCharacteristics1d', category=category, label=label, description=description)
        DefinitionBuilder.add_definition_to_class(self, definition=definition)

    def copy(self, name, category=None, label=None, description=None):
        return AntCharacteristics1dBuilder.build(
            df=self.df.copy(), name=name, category=category, label=label, description=description)


class AntCharacteristics1dBuilder:
    def __init__(self):
        pass

    @staticmethod
    def build(df, name, category=None, label=None, description=None):
        definition = DefinitionBuilder().build1d(
            name=name, category=category, object_type='AntCharacteristics1d',
            label=label, description=description
        )
        df.columns = [name]
        return AntCharacteristics1d(df.sort_index(), definition)
