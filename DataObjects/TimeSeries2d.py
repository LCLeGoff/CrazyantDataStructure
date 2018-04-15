import pandas as pd

from DataObjects.Definitions import DefinitionBuilder
from DataObjectBuilders.Builder2dDataObject import Builder2dDataObject
from DataObjectBuilders.BuilderExpAntFrameIndexedDataObject import BuilderExpAntFrameIndexedDataObject


class TimeSeries2d(Builder2dDataObject, BuilderExpAntFrameIndexedDataObject):
	def __init__(self, array, definition):
		Builder2dDataObject.__init__(self, array)
		BuilderExpAntFrameIndexedDataObject.__init__(self, array)
		DefinitionBuilder.build_from_definition(self, definition)


class TimeSeries2dBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build_from_1d(ts1, ts2, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
		array = pd.DataFrame(index=ts1.array.index)
		array[xname] = ts1.array
		array[yname] = ts2.array
		definition = DefinitionBuilder().build2d(
			name=name, category=category, object_type='TimeSeries2d',
			label=label, xlabel=xlabel, ylabel=ylabel, description=description
		)
		return TimeSeries2d(array, definition)

	@staticmethod
	def build_from_array(array, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
		array.columns = [xname, yname]
		definition = DefinitionBuilder().build2d(
			name=name, category=category, object_type='TimeSeries2d',
			label=label, xlabel=xlabel, ylabel=ylabel, description=description
		)
		return TimeSeries2d(array.sort_index(), definition)
