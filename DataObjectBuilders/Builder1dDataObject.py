from DataObjectBuilders.BuilderDataObject import BuilderDataObject


class Builder1dDataObject(BuilderDataObject):
	def __init__(self, array):
		if array.shape[1] != 1:
			raise ValueError('Shape not correct')
		else:
			BuilderDataObject.__init__(self, array)

	def get_values(self):
		return self.array[self.array.columns[0]]
