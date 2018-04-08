from DataObjects.Definitions import DefinitionBuilder
from Builders.DefinitionDataObjectBuilder import DefinitionDataObjectBuilder
from DataObjectBuilders.BuilderExpAntFrameIndexed1dDataObject import BuilderExpAntFrameIndexed1dDataObject


class Events(BuilderExpAntFrameIndexed1dDataObject):
	def __init__(self, array, definition):
		BuilderExpAntFrameIndexed1dDataObject.__init__(self, array)
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
