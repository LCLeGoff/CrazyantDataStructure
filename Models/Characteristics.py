from IndexedSeries.ExpIndexedMultiSeries import ExpIndexedMultiSeries
from IndexedSeries.ExpIndexedSeries import ExpIndexedSeries
from Builders.DefinitionObjectBuilder import DefinitionObjectBuilder


class Characteristics1d(ExpIndexedSeries):
	def __init__(self, array, definition):
		ExpIndexedSeries.__init__(self, array)
		DefinitionObjectBuilder.build(self, definition)


class Characteristics2d(ExpIndexedMultiSeries):
	def __init__(self, array, definition):
		ExpIndexedMultiSeries.__init__(self, array)
		DefinitionObjectBuilder.build(self, definition)
