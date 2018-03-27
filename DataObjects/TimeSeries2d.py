import pandas as pd

from DataObjects.Definitions import DefinitionBuilder
from Builders.DefinitionDataObjectBuilder import DefinitionDataObjectBuilder
from IndexedSeries.ExpAntFrameIndexed2dSeries import ExpAntFrameIndexed2dSeries


class TimeSeries2d(ExpAntFrameIndexed2dSeries):
	def __init__(self, array, definition):
		ExpAntFrameIndexed2dSeries.__init__(self, array)
		DefinitionDataObjectBuilder.build(self, definition)


class TimeSeries2dBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build(ts1, ts2, name, category, label, xlabel, ylabel, description):
		array = pd.DataFrame(index=ts1.array.index)
		array[ts1.name] = ts1.array
		array[ts2.name] = ts2.array
		definition = DefinitionBuilder().build2d(
			name=name, category=category, object_type='TimeSeries',
			label=label, xlabel=xlabel, ylabel=ylabel, description=description
		)
		return TimeSeries2d(array, definition)
