import pandas as pd


class TimeSeries1dRenamer:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def rename(self, ts, name, category=None, label=None, description=None):

        address = self.root + ts.category + '/TimeSeries.csv'
        df = pd.read_csv(address, index_col=['id_exp', 'id_ant', 'frame'])
        df.drop(ts.name, axis=1, inplace=True)

        ts.rename(name=name, category=category, label=label, description=description)

        df[name] = ts.df
        df.to_csv(address)
