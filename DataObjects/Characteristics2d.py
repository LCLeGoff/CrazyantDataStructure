from DataObjectBuilders.Builder2dDataObject import Builder2dDataObject
from DataObjectBuilders.BuilderExpIndexedDataObject import BuilderExpIndexedDataObject
from DataObjects.Definitions import DefinitionBuilder


class Characteristics2d(Builder2dDataObject, BuilderExpIndexedDataObject):
	def __init__(self, array, definition):
		BuilderExpIndexedDataObject.__init__(self, array)
		Builder2dDataObject.__init__(self, array)
		DefinitionBuilder.build_from_definition(self, definition)

	def copy(self, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
			return Characteristics2dBuilder.build_from_array(
				self.array.copy(), name=name, xname=xname, yname=yname,
				category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description)


class Characteristics2dBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build_from_array(array, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
		array.columns = [xname, yname]
		definition = DefinitionBuilder().build2d(
			name=name, category=category, object_type='Characteristics2d',
			label=label, xlabel=xlabel, ylabel=ylabel, description=description
		)
		return Characteristics2d(array, definition)
