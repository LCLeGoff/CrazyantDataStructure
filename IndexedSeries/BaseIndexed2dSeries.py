from IndexedSeries.BaseSeries import BaseSeries


class BaseIndexed2dSeries(BaseSeries):
	def __init__(self, array):
		BaseSeries.__init__(self, array)
