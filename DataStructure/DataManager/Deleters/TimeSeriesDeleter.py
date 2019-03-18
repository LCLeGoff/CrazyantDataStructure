import pandas as pd

from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name


class TimeSeriesDeleter:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def delete(self, ts):
        if ts.category == 'Raw':
            raise OSError('not allowed to delete TimeSeries of the category Raw')
        else:
            address = self.root + ts.category + '/TimeSeries.csv'
            df = pd.read_csv(address, index_col=[id_exp_name, id_ant_name, id_frame_name])
            df.drop(ts.name, axis=1, inplace=True)
            df.to_csv(address)
