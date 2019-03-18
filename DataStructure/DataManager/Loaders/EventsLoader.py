import pandas as pd

from DataStructure.DataObjects.Events1d import Events1d
from DataStructure.DataObjects.Events2d import Events2d
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name


class Events1dLoader:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def load(self, definition):
        add = self.root + definition.category + '/' + definition.name + '.csv'
        df = pd.read_csv(add)
        df.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)
        return Events1d(df, definition)


class Events2dLoader:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def load(self, definition):
        add = self.root + definition.category + '/' + definition.name + '.csv'
        return Events2d(pd.read_csv(add, index_col=[id_exp_name, id_ant_name, id_frame_name]), definition)
