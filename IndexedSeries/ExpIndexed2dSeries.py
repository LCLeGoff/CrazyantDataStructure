from IndexedSeries.BaseIndexed2dSeries import BaseIndexed2dSeries


class ExpIndexed2dSeries(BaseIndexed2dSeries):
	"""
	Class to deal with pandas object indexed by id_exp
	"""
	def __init__(self, array):
		BaseIndexed2dSeries.__init__(self, array)
		if array.index.names != ['id_exp']:
			raise IndexError('Index names are not id_exp')
		elif array.shape[1] != 2:
			raise ValueError('Shape not correct')

	def operation_on_columns(self, col_name, fct):
		"""
		Apply the lambda function fct to all values of columns col_name
		:param col_name: str
		:param fct: lambda function
		:return:
		"""
		self.array[col_name] = fct(self.array[col_name])

	def operation_on_id_exp(self, id_exp, fct):
		"""
		Apply a lambda function to the values associate to experiment id_exp
		:param id_exp: (int or list) experiment id on which the function is applied
		:param fct: function applied
		"""
		self.array.loc[id_exp, :, :] = fct(self.array.loc[id_exp])
