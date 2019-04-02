import numpy as np

from DataStructure.DataObjectBuilders.BuilderDataObject import BuilderDataObject


class Builder2dDataObject(BuilderDataObject):
    def __init__(self, df):
        if df.shape[1] != 2:
            raise ValueError('Shape not correct')
        else:
            BuilderDataObject.__init__(self, df)
            self.name_col = self.df.columns

    def rename_df(self, xname, yname):
        self.df.columns = [xname, yname]

    def get_x_values(self):
        return self.df[self.df.columns[0]]

    def get_y_values(self):
        return self.df[self.df.columns[1]]

    def get_value(self, idx):
        return self.df.loc[idx, self.df.columns]

    def get_array(self):
        return np.array(list(zip(self.get_x_values(), self.get_y_values())))

    def replace_values(self, name_columns, vals):
        self.df[name_columns] = vals
