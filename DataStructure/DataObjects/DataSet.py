from DataStructure.DataObjects.Definitions import DefinitionBuilder
from DataStructure.DataSetDecorators.IndexedDataSetDecorator import IndexedDataSetDecorator
from DataStructure.VariableNames import dataset_name


class DataSet(IndexedDataSetDecorator):

    def __init__(self, df, definition):
        IndexedDataSetDecorator.__init__(self, df)
        DefinitionBuilder.add_definition_to_class(self, definition)

    def rename(self, name, category=None, label=None, description=None):

        if category is None:
            category = self.category
        if label is None:
            label = self.label
        if description is None:
            description = self.description

        definition = DefinitionBuilder().build_dataset(name=name, object_type=dataset_name, category=category,
                                                       label=label, description=description, nb_indexes=self.nb_indexes)

        DefinitionBuilder.add_definition_to_class(self, definition=definition)

    def copy(self, name, category=None, label=None, description=None):
        return DataSetBuilder.build(df=self.df.copy(), name=name, category=category, label=label,
                                    description=description, nb_indexes=self.nb_indexes)


class DataSetBuilder:
    def __init__(self):
        pass

    @staticmethod
    def build(df, name, category=None, label=None, description=None, nb_indexes=None):

        definition = DefinitionBuilder().build_dataset(name=name, category=category, object_type=dataset_name,
                                                       label=label, description=description, nb_indexes=nb_indexes)
        return DataSet(df.sort_index(), definition)
