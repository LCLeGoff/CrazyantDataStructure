from DataObjectBuilders.BuilderDataObject import BuilderDataObject


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
