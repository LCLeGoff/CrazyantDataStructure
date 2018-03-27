import pandas as pd

from DataObjects.Definitions import DefinitionBuilder
from Builders.DefinitionDataObjectBuilder import DefinitionDataObjectBuilder
from IndexedSeries.ExpAntFrameIndexed2dSeries import ExpAntFrameIndexed2dSeries


class Events2d(ExpAntFrameIndexed2dSeries):
	def __init__(self, array, definition):
		ExpAntFrameIndexed2dSeries.__init__(self, array)
		DefinitionDataObjectBuilder.build(self, definition)


class Events2dBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build(event1, event2, name, category, label, xlabel, ylabel, description):
		array = pd.DataFrame(index=event1.array.index)
		array[event1.name] = event1.array
		array[event2.name] = event2.array
		definition = DefinitionBuilder().build2d(
			name=name, category=category, object_type='Events2d',
			label=label, xlabel=xlabel, ylabel=ylabel, description=description
		)
		return Events2d(array, definition)
