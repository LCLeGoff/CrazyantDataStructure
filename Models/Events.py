from Builders.CharacteristicObjectBuilder import CharacteristicObjectBuilder
from IndexedSeries.ExpAntFrameIndexedSeries import ExpAntFrameIndexedSeries


class Events(ExpAntFrameIndexedSeries):
	def __init__(self, array, name):
		ExpAntFrameIndexedSeries.__init__(self, array)
		CharacteristicObjectBuilder.build(self, name)
