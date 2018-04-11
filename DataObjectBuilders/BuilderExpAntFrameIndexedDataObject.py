
class BuilderExpAntFrameIndexedDataObject:
	def __init__(self, array):
		if array.index.names != ['id_exp', 'id_ant', 'frame']:
			raise IndexError('Index names are not (id_exp, id_ant, frame)')
		else:
			self.array = array

	def operation_on_id_exp(self, id_exp, fct):
		self.array.loc[id_exp, :, :] = fct(self.array.loc[id_exp, :, :])
