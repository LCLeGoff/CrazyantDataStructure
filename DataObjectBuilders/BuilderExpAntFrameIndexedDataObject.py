from DataObjectBuilders.BuilderDataObject import BuilderDataObject


class BuilderExpAntFrameIndexedDataObject(BuilderDataObject):
	def __init__(self, array):
		BuilderDataObject.__init__(self, array)
		if array.index.names != ['id_exp', 'id_ant', 'frame']:
			raise IndexError('Index names are not (id_exp, id_ant, frame)')
		else:
			self.name_col = self.array.columns[0]

	def operation_on_id_exp(self, id_exp, fct):
		self.array.loc[id_exp, :, :] = fct(self.array.loc[id_exp, :, :])
