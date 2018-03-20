from TimeSeries import TimeSeries


class TimeSeriesBuilder:

	@staticmethod
	def build(tab, name):
		return TimeSeries(tab, name)
