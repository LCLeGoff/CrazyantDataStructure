
from DataObjects.Definitions import DefinitionBuilder
from DataObjectBuilders.Builder1dDataObject import Builder1dDataObject
from DataObjectBuilders.BuilderExpAntFrameIndexedDataObject import BuilderExpAntFrameIndexedDataObject


class Events1d(Builder1dDataObject, BuilderExpAntFrameIndexedDataObject):
	def __init__(self, array, definition):
		Builder1dDataObject.__init__(self, array)
		BuilderExpAntFrameIndexedDataObject.__init__(self, array)
		DefinitionBuilder.build_from_definition(self, definition)

	def copy(self, name, category=None, label=None, description=None):
			return Events1dBuilder.build(
				array=self.array.copy(), name=name, category=category, label=label, description=description)


class Events1dBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build(array, name, category=None, label=None, description=None):
		definition = DefinitionBuilder().build1d(
			name=name, category=category, object_type='Events',
			label=label, description=description
		)
		return Events1d(array.sort_index(), definition)
