from IndexedSeries.BaseSeries import BaseSeries


class BaseIndexedSeries(BaseSeries):
	def __init__(self, array):
		BaseSeries.__init__(self, array)

	def get_values(self):
		return self.array[self.array.columns[0]]
