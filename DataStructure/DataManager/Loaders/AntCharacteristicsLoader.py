import pandas as pd

from DataStructure.DataObjects.AntCharacteristics1d import AntCharacteristics1d


class AntCharacteristics1dLoader:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def load(self, definition):
        address = self.root + definition.category + '/' + definition.name + '.csv'
        df = pd.read_csv(address, index_col=['id_exp', 'id_ant'])
        return AntCharacteristics1d(df, definition)
