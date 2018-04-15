from DataObjectBuilders.Builder1dDataObject import Builder1dDataObject
from DataObjectBuilders.BuilderExpIndexedDataObject import BuilderExpIndexedDataObject
from DataObjects.Definitions import DefinitionBuilder


class Characteristics1d(Builder1dDataObject, BuilderExpIndexedDataObject):
	def __init__(self, array, definition):
		Builder1dDataObject.__init__(self, array)
		BuilderExpIndexedDataObject.__init__(self, array)
		DefinitionBuilder.build_from_definition(self, definition)

	def copy(self, name, category=None, label=None, description=None):
			return Characteristics1dBuilder.build(
				array=self.array.copy(), name=name, category=category, label=label, description=description)


class Characteristics1dBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build(array, name, category=None, label=None, description=None):
		definition = DefinitionBuilder().build1d(
			name=name, category=category, object_type='Characteristics1d',
			label=label, description=description
		)
		return Characteristics1d(array.sort_index(), definition)
