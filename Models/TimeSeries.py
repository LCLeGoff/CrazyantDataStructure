from Builders.CharacteristicObjectBuilder import CharacteristicObjectBuilder
from IndexedSeries.ExpAntFrameIndexedSeries import ExpAntFrameIndexedSeries


class TimeSeries(ExpAntFrameIndexedSeries):
	def __init__(self, array, name):
		ExpAntFrameIndexedSeries.__init__(self, array)
		CharacteristicObjectBuilder.build(self, name)

	def operation_with_characteristics1d(self, chara, fct):
		self.array[self.name_col] = fct(self.array[self.name_col], chara.array[chara.name_col])

	def operation_with_characteristics2d(self, chara, name_col, fct):
		self.array[self.name_col] = fct(self.array[self.name_col], chara.array[name_col])

