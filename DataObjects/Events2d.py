import pandas as pd

from DataObjects.Definitions import DefinitionBuilder
from Builders.DefinitionDataObjectBuilder import DefinitionDataObjectBuilder
from DataObjectBuilders.BuilderExpAntFrameIndexed2dDataObject import BuilderExpAntFrameIndexed2dDataObject


class Events2d(BuilderExpAntFrameIndexed2dDataObject):
	def __init__(self, array, definition):
		BuilderExpAntFrameIndexed2dDataObject.__init__(self, array)
		DefinitionDataObjectBuilder.build(self, definition)

	def copy(self, name, xname, yname, category, label, xlabel, ylabel, description):
			return Events2dBuilder.build_from_array(self.array, name, xname, yname, category, label, xlabel, ylabel, description)


class Events2dBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build(event1, event2, name, xname, yname, category, label, xlabel, ylabel, description):
		array = pd.DataFrame(index=event1.array.index)
		array[xname] = event1.array
		array[yname] = event2.array
		definition = DefinitionBuilder().build2d(
			name=name, category=category, object_type='Events2d',
			label=label, xlabel=xlabel, ylabel=ylabel, description=description
		)
		return Events2d(array, definition)

	@staticmethod
	def build_from_array(array, name, xname, yname, category, label, xlabel, ylabel, description):
		array.columns = [xname, yname]
		definition = DefinitionBuilder().build2d(
			name=name, category=category, object_type='Events2d',
			label=label, xlabel=xlabel, ylabel=ylabel, description=description
		)
		return Events2d(array, definition)
