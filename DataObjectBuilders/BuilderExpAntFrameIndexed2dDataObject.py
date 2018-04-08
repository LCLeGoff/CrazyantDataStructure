from DataObjectBuilders.Builder2dDataObject import Builder2dDataObject


class BuilderExpAntFrameIndexed2dDataObject(Builder2dDataObject):
	"""
	Class to deal with pandas object indexed by id_exp
	"""
	def __init__(self, array):
		Builder2dDataObject.__init__(self, array)
		if array.index.names != ['id_exp', 'id_ant', 'frame']:
			raise IndexError('Index names are not (id_exp, id_ant, frame)')
		elif array.shape[1] != 2:
			raise ValueError('Shape not correct')
		else:
			self.name_col = self.array.columns[0]

	def operation_on_id_exp(self, id_exp, fct, inplace=True):
		"""
		Apply a lambda function to the values associate to experiment id_exp
		:param id_exp: (int or list) experiment id on which the function is applied
		:param fct: function applied
		:param inplace: if True, change the original pandas object, if False return a new pandas object
		"""
		if inplace is True:
			self.array.loc[id_exp, :, :] = fct(self.array.loc[id_exp, :, :])
		else:
			return fct(self.array.loc[id_exp, :, :])
