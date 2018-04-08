from DataObjects.Definitions import DefinitionBuilder
from Builders.DefinitionDataObjectBuilder import DefinitionDataObjectBuilder
from IndexedSeries.ExpAntFrameIndexedSeries import ExpAntFrameIndexedSeries


class Events(ExpAntFrameIndexedSeries):
	def __init__(self, array, definition):
		ExpAntFrameIndexedSeries.__init__(self, array)
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
