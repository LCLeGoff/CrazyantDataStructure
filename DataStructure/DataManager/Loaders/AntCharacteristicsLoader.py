import pandas as pd

from DataStructure.DataObjects.AntCharacteristics1d import AntCharacteristics1d
from DataStructure.VariableNames import id_exp_name, id_ant_name


class AntCharacteristics1dLoader:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def load(self, definition):
        address = self.root + definition.category + '/' + definition.name + '.csv'
        df = pd.read_csv(address, index_col=[id_exp_name, id_ant_name])
        return AntCharacteristics1d(df, definition)
