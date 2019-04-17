import pandas as pd

from DataStructure.DataObjects.CharacteristicEvents1d import CharacteristicEvents1d
from DataStructure.DataObjects.CharacteristicEvents2d import CharacteristicEvents2d
from DataStructure.VariableNames import id_exp_name, id_frame_name


class CharacteristicEvents1dLoader:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def load(self, definition):
        add = self.root + definition.category + '/DataSets/' + definition.name + '.csv'
        df = pd.read_csv(add)
        df.set_index([id_exp_name, id_frame_name], inplace=True)
        return CharacteristicEvents1d(df, definition)


class CharacteristicEvents2dLoader:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def load(self, definition):
        add = self.root + definition.category + '/DataSets/' + definition.name + '.csv'
        return CharacteristicEvents2d(pd.read_csv(add, index_col=[id_exp_name, id_frame_name]), definition)
