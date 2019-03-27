import pandas as pd

from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name


class TimeSeries1dRenamer:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def rename(self, ts, name, category=None, label=None, description=None):

        address = self.root + ts.category + '/TimeSeries.csv'
        if category is None:
            new_address = address
        else:
            new_address = self.root + category + '/TimeSeries.csv'

        df = pd.read_csv(address, index_col=[id_exp_name, id_ant_name, id_frame_name])
        df.drop(ts.name, axis=1, inplace=True)
        df.to_csv(address)

        df = pd.read_csv(new_address, index_col=[id_exp_name, id_ant_name, id_frame_name])
        ts.rename(name=name, category=category, label=label, description=description)

        df[name] = ts.df
        df.to_csv(new_address)
