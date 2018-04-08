from DataObjectBuilders.BuilderExpIndexed2dDataObject import BuilderExpIndexed2dDataObject
from DataObjectBuilders.BuilderExpIndexed1dDataObject import BuilderExpIndexed1dDataObject
from Builders.DefinitionDataObjectBuilder import DefinitionDataObjectBuilder


class Characteristics1d(BuilderExpIndexed1dDataObject):
	def __init__(self, array, definition):
		BuilderExpIndexed1dDataObject.__init__(self, array)
		DefinitionDataObjectBuilder.build(self, definition)


class Characteristics2d(BuilderExpIndexed2dDataObject):
	def __init__(self, array, definition):
		BuilderExpIndexed2dDataObject.__init__(self, array)
		DefinitionDataObjectBuilder.build(self, definition)
