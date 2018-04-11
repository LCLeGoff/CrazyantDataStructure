import numpy as np

from DataObjects.Definitions import DefinitionBuilder
from DataObjectBuilders.Builder1dDataObject import Builder1dDataObject
from DataObjectBuilders.BuilderExpAntFrameIndexedDataObject import BuilderExpAntFrameIndexedDataObject
from DataObjects.Events1d import Events1dBuilder


class TimeSeries1d(Builder1dDataObject, BuilderExpAntFrameIndexedDataObject):
	def __init__(self, array, definition):
		# TODO: Fix this conception mistake
		array.columns = [definition.name]
		Builder1dDataObject.__init__(self, array)
		BuilderExpAntFrameIndexedDataObject.__init__(self, array)
		DefinitionBuilder.build_from_definition(self, definition)

	def copy(self, name, category=None, label=None, description=None):
			return TimeSeries1dBuilder.build(
				array=self.array.copy(), name=name, category=category, label=label, description=description)

	def extract_event(self, name, category=None, label=None, description=None):
		ts_array = self.get_array()
		ts_array2 = ts_array[1:]-ts_array[:-1]
		mask = [0]+list(np.where(ts_array2 != 0)[0]+1)
		event = self.array.iloc[mask]
		event.columns = [name]
		return Events1dBuilder().build(array=event, name=name, category=category, label=label, description=description)


class TimeSeries1dBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build(array, name, category=None, label=None, description=None):
		definition = DefinitionBuilder().build1d(
			name=name, category=category, object_type='TimeSeries',
			label=label, description=description
		)
		return TimeSeries1d(array.copy(), definition)
