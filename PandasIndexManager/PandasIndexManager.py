class PandasIndexManager:
	def __init__(self):
		pass

	@staticmethod
	def get_id_exp_index(array):
		arr = array.reset_index()
		if 'id_exp' not in arr.columns:
			raise IndexError('array does not have id_exp as index')
		else:
			return set([idx for idx in array.reset_index().set_index(['id_exp']).index])
