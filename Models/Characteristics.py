from IndexedSeries.ExpIndexedMultiSeries import ExpIndexedMultiSeries
from IndexedSeries.ExpIndexedSeries import ExpIndexedSeries
from Builders.CharacteristicObjectBuilder import CharacteristicObjectBuilder


class Characteristics1d(ExpIndexedSeries):
	def __init__(self, array, name):
		ExpIndexedSeries.__init__(self, array)
		CharacteristicObjectBuilder.build(self, name)


class Characteristics2d(ExpIndexedMultiSeries):
	def __init__(self, array, name):
		ExpIndexedMultiSeries.__init__(self, array)
		CharacteristicObjectBuilder.build(self, name)
