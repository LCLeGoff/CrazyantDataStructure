class BaseSeries:
	def __init__(self, array):
		self.array = array

	def operation(self, fct):
		"""
		Apply a lambda function to all values
		:param fct: function applied
		"""
		self.array = fct(self.array)
