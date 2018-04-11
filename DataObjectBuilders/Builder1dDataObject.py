from DataObjectBuilders.BuilderDataObject import BuilderDataObject
import numpy as np


class Builder1dDataObject(BuilderDataObject):
	def __init__(self, array):
		if array.shape[1] != 1:
			raise ValueError('Shape not correct')
		else:
			BuilderDataObject.__init__(self, array)
			self.name_col = self.array.columns[0]

	def get_values(self):
		return self.array[self.array.columns[0]]

	def get_value(self, idx):
		return self.array.loc[idx, self.array.columns[0]]

	def get_array(self):
		return np.array(self.array[self.array.columns[0]])

	def replace_values(self, vals):
		self.array[self.array.columns[0]] = vals

	def operation_with_data1d(self, obj, fct):
		self.replace_values(fct(self.array[self.name_col], obj.array[obj.name_col]))

	def operation_with_data2d(self, obj, obj_name_col, fct):
		self.replace_values(fct(self.array[self.name_col], obj.array[obj_name_col]))
