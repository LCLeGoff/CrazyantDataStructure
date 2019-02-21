import pandas as pd

from DataStructure.DataObjects.Events1d import Events1d
from DataStructure.DataObjects.Events2d import Events2d


class Events1dLoader:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def load(self, definition):
        add = self.root + definition.category + '/' + definition.name + '.csv'
        df = pd.read_csv(add)
        df.set_index(['id_exp', 'id_ant', 'frame'], inplace=True)
        return Events1d(df, definition)


class Events2dLoader:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def load(self, definition):
        add = self.root + definition.category + '/' + definition.name + '.csv'
        return Events2d(pd.read_csv(add, index_col=['id_exp', 'id_ant', 'frame']), definition)
