import pandas as pd

from DataStructure.VariableNames import id_exp_name, id_frame_name


class CharacteristicTimeSeries1dRenamer:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def rename(self, chara_ts1d, name, category=None, label=None, description=None):

        address = self.root + chara_ts1d.category + '/CharacteristicTimeSeries.csv'
        if category is None:
            new_address = address
        else:
            new_address = self.root + category + '/CharacteristicTimeSeries.csv'

        df = pd.read_csv(address, index_col=[id_exp_name, id_frame_name])
        df.drop(chara_ts1d.name, axis=1, inplace=True)
        df.to_csv(address)

        df = pd.read_csv(new_address, index_col=[id_exp_name, id_frame_name])
        chara_ts1d.rename(name=name, category=category, label=label, description=description)

        df[name] = chara_ts1d.df
        df.to_csv(new_address)
