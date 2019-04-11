import pandas as pd

from DataStructure.DataObjects.DataSet import DataSet


class DataSetLoader:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def load(self, definition):

        add = self.root + definition.category + '/DataSets/' + definition.name + '.csv'
        df = pd.read_csv(add)
        index_names = list(df.columns[:definition.nb_indexes])
        df.set_index(index_names, inplace=True)

        return DataSet(df, definition)
