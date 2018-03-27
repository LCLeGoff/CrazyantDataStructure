from Builders.DefinitionObjectBuilder import DefinitionObjectBuilder
from IndexedSeries.ExpAntFrameIndexedSeries import ExpAntFrameIndexedSeries


class Events(ExpAntFrameIndexedSeries):
	def __init__(self, array, definition):
		ExpAntFrameIndexedSeries.__init__(self, array)
		DefinitionObjectBuilder.build(self, definition)
