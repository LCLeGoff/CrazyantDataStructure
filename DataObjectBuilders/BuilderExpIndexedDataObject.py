from DataObjectBuilders.BuilderDataObject import BuilderDataObject


class BuilderExpIndexedDataObject(BuilderDataObject):
	def __init__(self, array):
		BuilderDataObject.__init__(self, array)
		if array.index.names != ['id_exp']:
			raise IndexError('Index names are not id_exp')
		else:
			self.array = array
			self.name_col = self.array.columns[0]

	def operation_on_id_exp(self, id_exp, fct):
		self.array.loc[id_exp] = fct(self.array.loc[id_exp])
