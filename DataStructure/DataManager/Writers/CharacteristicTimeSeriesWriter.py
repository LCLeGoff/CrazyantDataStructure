import pandas as pd


class CharacteristicTimeSeries1dWriter:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def write(self, ts):
        if ts.category == 'Raw':
            raise OSError('not allowed to modify CharacteristicTimeSeries of the category Raw')
        else:
            add = self.root + ts.category + '/CharacteristicTimeSeries.csv'
            df = pd.read_csv(add, index_col=['id_exp', 'frame'])
            df[ts.name] = ts.df
            df.to_csv(add)
