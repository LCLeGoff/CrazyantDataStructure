from BaseSeries import BaseSeries


class ExpAntFrameIndexedSeries(BaseSeries):
	"""
	Class to deal with pandas object indexed by id_exp
	"""
	def __init__(self, array):
		BaseSeries.__init__(self, array)
		if array.index.names != ['id_exp', 'id_ant', 'frame']:
			raise IndexError('Index names are not (id_exp, id_ant, frame)')
		elif array.shape[1] != 1:
			raise ValueError('Shape not correct')

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
