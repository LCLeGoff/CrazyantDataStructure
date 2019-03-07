from DataStructure.DataObjects.Definitions import DefinitionBuilder
from Tools.Plotter.Plotter1d import Plotter1d


class DataSet(OneIndexedDataSet):
    def __init__(self, df, definition):
        OneIndexedDataSet.__init__(self, df)
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
            name=name, object_type='Events1d', category=category, label=label, description=description)
        DefinitionBuilder.add_definition_to_class(self, definition=definition)

    def copy(self, name, category=None, label=None, description=None):
        return DataSetBuilder.build(
            df=self.df.copy(), name=name, category=category, label=label, description=description)


class DataSetBuilder:
    def __init__(self):
        pass

    @staticmethod
    def build(df, name, category=None, label=None, description=None):
        definition = DefinitionBuilder().build1d(
            name=name, category=category, object_type='Events1d',
            label=label, description=description
        )
        df.columns = [name]
        return DataSet(df.sort_index(), definition)
