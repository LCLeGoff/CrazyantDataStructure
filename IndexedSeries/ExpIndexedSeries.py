from IndexedSeries.BaseIndexedSeries import BaseSeries


class ExpIndexedSeries(BaseSeries):
	"""
	Class to deal with pandas object indexed by (id_exp, id_ant, frame)
	"""
	def __init__(self, array):
		BaseSeries.__init__(self, array)
		if array.index.names != ['id_exp']:
			raise IndexError('Index names are not id_exp')
		elif array.shape[1] != 1:
			raise ValueError('Shape not correct')
		else:
			self.array = array
			self.name_col = self.array.columns[0]

	def operation_on_id_exp(self, id_exp, fct, inplace=True):
		"""
		Apply a lambda function to the values associate to experiment id_exp
		:param id_exp: (int or list) experiment id on which the function is applied
		:param fct: function applied
		:param inplace: if True, change the original pandas object, if False return a new pandas object
		"""
		if inplace is True:
			self.array.loc[id_exp, :, :] = fct(self.array.loc[id_exp])
		else:
			return fct(self.array.loc[id_exp])
