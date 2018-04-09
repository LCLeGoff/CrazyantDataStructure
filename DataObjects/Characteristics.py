from DataObjectBuilders.Builder1dDataObject import Builder1dDataObject
from DataObjectBuilders.Builder2dDataObject import Builder2dDataObject
from DataObjectBuilders.BuilderExpIndexedDataObject import BuilderExpIndexedDataObject
from Builders.DefinitionDataObjectBuilder import DefinitionDataObjectBuilder


class Characteristics1d(Builder1dDataObject, BuilderExpIndexedDataObject):
	def __init__(self, array, definition):
		BuilderExpIndexedDataObject.__init__(self, array)
		Builder1dDataObject.__init__(self, array)
		DefinitionDataObjectBuilder.build(self, definition)


class Characteristics2d(Builder2dDataObject, BuilderExpIndexedDataObject):
	def __init__(self, array, definition):
		BuilderExpIndexedDataObject.__init__(self, array)
		Builder2dDataObject.__init__(self, array)
		DefinitionDataObjectBuilder.build(self, definition)
