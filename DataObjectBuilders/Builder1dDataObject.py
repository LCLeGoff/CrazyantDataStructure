from DataObjectBuilders.BuilderDataObject import BuilderDataObject
import numpy as np


class Builder1dDataObject(BuilderDataObject):
	def __init__(self, array):
		if array.shape[1] != 1:
			raise ValueError('Shape not correct')
		else:
			BuilderDataObject.__init__(self, array)

	def get_values(self):
		return self.array[self.array.columns[0]]

	def get_array(self):
		return np.array(self.array[self.array.columns[0]])

	def replace_values(self, vals):
		self.array[self.array.columns[0]] = vals
