from PandasIndexManager.PandasIndexManager import PandasIndexManager


class BuilderDataObject:
	def __init__(self, array):
		self.array = array

	def operation(self, fct):
		"""
		Apply a lambda function to all values
		:param fct: function applied
		"""
		self.array = fct(self.array)

	def print(self, short=True):
		if short:
			print(self.array.head())
		else:
			print(self.array)

	def get_value(self, idx):
		return self.array.loc[idx, :]

	def get_id_exp_index(self):
		return PandasIndexManager().get_id_exp_index(self.array)
