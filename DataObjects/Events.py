from DataObjects.Definitions import DefinitionBuilder
from Builders.DefinitionDataObjectBuilder import DefinitionDataObjectBuilder
from DataObjectBuilders.Builder1dDataObject import Builder1dDataObject
from DataObjectBuilders.BuilderExpAntFrameIndexedDataObject import BuilderExpAntFrameIndexedDataObject


class Events(Builder1dDataObject, BuilderExpAntFrameIndexedDataObject):
	def __init__(self, array, definition):
		Builder1dDataObject.__init__(self, array)
		BuilderExpAntFrameIndexedDataObject.__init__(self, array)
		DefinitionDataObjectBuilder.build(self, definition)

	def copy(self, name, category, label, description):
			return EventsBuilder.build(self.array, name, category, label, description)


class EventsBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build(array, name, category, label, description):
		definition = DefinitionBuilder().build1d(
			name=name, category=category, object_type='Events',
			label=label, description=description
		)
		return Events(array.copy(), definition)
