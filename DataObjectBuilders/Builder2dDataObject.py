from DataObjectBuilders.BuilderDataObject import BuilderDataObject
import numpy as np


class Builder2dDataObject(BuilderDataObject):
	def __init__(self, array):
		if array.shape[1] != 2:
			raise ValueError('Shape not correct')
		else:
			BuilderDataObject.__init__(self, array)

	def get_x_values(self):
		return self.array[self.array.columns[0]]

	def get_y_values(self):
		return self.array[self.array.columns[1]]

	def get_array(self):
		return np.array(list(zip(self.get_x_values(), self.get_y_values())))

	def replace_values(self, name_columns, vals):
		self.array[name_columns] = vals
