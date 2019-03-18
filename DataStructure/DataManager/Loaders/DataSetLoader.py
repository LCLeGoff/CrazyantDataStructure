import pandas as pd

from DataStructure.DataObjects.DataSet import DataSet


class DataSetLoader:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def load(self, definition):
        add = self.root + definition.category + '/' + definition.name + '.csv'
        df = pd.read_csv(add)
        index_name = [df.columns[0]]
        df.set_index(index_name, inplace=True)
        return DataSet(df, definition)
