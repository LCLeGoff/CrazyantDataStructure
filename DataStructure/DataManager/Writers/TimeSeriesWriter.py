import pandas as pd

from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name


class TimeSeriesWriter:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def write(self, ts):
        if ts.category == 'Raw':
            raise OSError('not allowed to modify TimeSeries of the category Raw')
        else:
            add = self.root + ts.category + '/TimeSeries.csv'
            df = pd.read_csv(add, index_col=[id_exp_name, id_ant_name, id_frame_name])
            type_is_int64 = True
            for index_name in df.index.names:
                type_is_int64 *= df.index.get_level_values(index_name).dtype == 'int64'

            if type_is_int64:

                df[ts.name] = ts.df
                df.to_csv(add)
            else:
                raise TypeError('Index are not int')
