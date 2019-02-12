from DataStructure.DataObjectBuilders.Builder2dDataObject import Builder2dDataObject
from DataStructure.DataObjectBuilders.BuilderExpIndexedDataObject import BuilderExpIndexedDataObject
from DataStructure.DataObjects.Definitions import DefinitionBuilder


class Characteristics2d(Builder2dDataObject, BuilderExpIndexedDataObject):
    def __init__(self, df, definition):
        BuilderExpIndexedDataObject.__init__(self, df)
        Builder2dDataObject.__init__(self, df)
        DefinitionBuilder.add_definition_to_class(self, definition)

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
            name=name, object_type='Characteristics2d', category=category,
            label=label, xlabel=xlabel, ylabel=ylabel, description=description)
        DefinitionBuilder.add_definition_to_class(self, definition=definition)

    def copy(self, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
        return Characteristics2dBuilder.build_from_df(
            self.df.copy(), name=name, xname=xname, yname=yname,
            category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description)


class Characteristics2dBuilder:
    def __init__(self):
        pass

    @staticmethod
    def build_from_df(df, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
        df.columns = [xname, yname]
        definition = DefinitionBuilder().build2d(
            name=name, category=category, object_type='Characteristics2d',
            label=label, xlabel=xlabel, ylabel=ylabel, description=description
        )
        return Characteristics2d(df.sort_index(), definition)
