from DataStructure.DataObjectBuilders.Builder1dDataObject import Builder1dDataObject
from DataStructure.DataObjectBuilders.BuilderExpIndexedDataObject import BuilderExpIndexedDataObject
from DataStructure.DataObjects.Definitions import DefinitionBuilder


class Characteristics1d(Builder1dDataObject, BuilderExpIndexedDataObject):
    def __init__(self, df, definition):
        Builder1dDataObject.__init__(self, df)
        BuilderExpIndexedDataObject.__init__(self, df)
        DefinitionBuilder.build_from_definition(self, definition)

    def copy(self, name, category=None, label=None, description=None):
        return Characteristics1dBuilder.build(
            df=self.df.copy(), name=name, category=category, label=label, description=description)


class Characteristics1dBuilder:
    def __init__(self):
        pass

    @staticmethod
    def build(df, name, category=None, label=None, description=None):
        definition = DefinitionBuilder().build1d(
            name=name, category=category, object_type='Characteristics1d',
            label=label, description=description
        )
        df.columns = [name]
        return Characteristics1d(df.sort_index(), definition)
