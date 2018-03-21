from ObjectClassBuilder import ObjectClassBuilder


class TimeSeries:
	def __init__(self, array, name):
		ObjectClassBuilder.build(self, array, name)
