import pandas as pd

from DataObjects.Definitions import DefinitionBuilder
from DataObjectBuilders.Builder2dDataObject import Builder2dDataObject
from DataObjectBuilders.BuilderExpAntFrameIndexedDataObject import BuilderExpAntFrameIndexedDataObject


class Events2d(Builder2dDataObject, BuilderExpAntFrameIndexedDataObject):
	def __init__(self, array, definition):
		Builder2dDataObject.__init__(self, array)
		BuilderExpAntFrameIndexedDataObject.__init__(self, array)
		DefinitionBuilder.build_from_definition(self, definition)

	def copy(self, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
			return Events2dBuilder.build_from_array(
				self.array.copy(), name=name, xname=xname, yname=yname,
				category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description)


class Events2dBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build(event1, event2, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
		array = pd.DataFrame(index=event1.array.index)
		array[xname] = event1.array
		array[yname] = event2.array
		definition = DefinitionBuilder().build2d(
			name=name, category=category, object_type='Events2d',
			label=label, xlabel=xlabel, ylabel=ylabel, description=description
		)
		return Events2d(array, definition)

	@staticmethod
	def build_from_array(array, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
		array.columns = [xname, yname]
		definition = DefinitionBuilder().build2d(
			name=name, category=category, object_type='Events2d',
			label=label, xlabel=xlabel, ylabel=ylabel, description=description
		)
		return Events2d(array, definition)
