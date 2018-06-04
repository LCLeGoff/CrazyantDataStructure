import numpy as np

from DataStructure.DataObjectBuilders.BuilderDataObject import BuilderDataObject


class Builder1dDataObject(BuilderDataObject):
    def __init__(self, df):
        if df.shape[1] != 1:
            raise ValueError('Shape not correct')
        else:
            BuilderDataObject.__init__(self, df)
            self.name_col = self.df.columns[0]

    def get_values(self):
        return self.df[self.df.columns[0]]

    def get_value(self, idx):
        return self.df.loc[idx, self.df.columns[0]]

    def get_array(self):
        return np.array(self.df[self.df.columns[0]])

    def replace_values(self, vals):
        self.df[self.df.columns[0]] = vals

    def operation_with_data1d(self, obj, fct):
        self.replace_values(fct(self.df[self.name_col], obj.df[obj.name_col]))

    def operation_with_data2d(self, obj, obj_name_col, fct):
        self.replace_values(fct(self.df[self.name_col], obj.df[obj_name_col]))
