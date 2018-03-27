from IndexedSeries.ExpIndexed2dSeries import ExpIndexed2dSeries
from IndexedSeries.ExpIndexedSeries import ExpIndexedSeries
from Builders.DefinitionDataObjectBuilder import DefinitionDataObjectBuilder


class Characteristics1d(ExpIndexedSeries):
	def __init__(self, array, definition):
		ExpIndexedSeries.__init__(self, array)
		DefinitionDataObjectBuilder.build(self, definition)


class Characteristics2d(ExpIndexed2dSeries):
	def __init__(self, array, definition):
		ExpIndexed2dSeries.__init__(self, array)
		DefinitionDataObjectBuilder.build(self, definition)
