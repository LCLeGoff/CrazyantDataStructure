import numpy as np

from DataStructure.DataObjectBuilders.BuilderDataObject import BuilderDataObject


class Builder2dDataObject(BuilderDataObject):
    def __init__(self, df):
        if df.shape[1] != 2:
            raise ValueError('Shape not correct')
        else:
            BuilderDataObject.__init__(self, df)

    def get_x_values(self):
        return self.df[self.df.columns[0]]

    def get_y_values(self):
        return self.df[self.df.columns[1]]

    def get_array(self):
        return np.array(list(zip(self.get_x_values(), self.get_y_values())))

    def replace_values(self, name_columns, vals):
        self.df[name_columns] = vals

    def compute_delta(self, name=None, filter_obj=None):
        if name is None:
            name = 'delta_'+self.df.columns[0]

        delta_df = self.pandas_index_manager.create_empty_exp_ant_frame_indexed_2d_df(name)

        self.fill_delta_df(delta_df, filter_obj)

        return delta_df
