import pandas as pd

from DataObjects.Definitions import DefinitionBuilder
from Builders.DefinitionDataObjectBuilder import DefinitionDataObjectBuilder
from DataObjectBuilders.BuilderExpAntFrameIndexed2dDataObject import BuilderExpAntFrameIndexed2dDataObject


class TimeSeries2d(BuilderExpAntFrameIndexed2dDataObject):
	def __init__(self, array, definition):
		BuilderExpAntFrameIndexed2dDataObject.__init__(self, array)
		DefinitionDataObjectBuilder.build(self, definition)


class TimeSeries2dBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build(ts1, ts2, name, xname, yname, category, label, xlabel, ylabel, description):
		array = pd.DataFrame(index=ts1.array.index)
		array[xname] = ts1.array
		array[yname] = ts2.array
		definition = DefinitionBuilder().build2d(
			name=name, category=category, object_type='TimeSeries2d',
			label=label, xlabel=xlabel, ylabel=ylabel, description=description
		)
		return TimeSeries2d(array, definition)
