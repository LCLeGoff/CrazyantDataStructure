from DataObjectBuilders.Builder1dDataObject import Builder1dDataObject
from DataObjectBuilders.BuilderExpAntIndexedDataObject import BuilderExpAntIndexedDataObject
from DataObjects.Definitions import DefinitionBuilder


class AntCharacteristics1d(Builder1dDataObject, BuilderExpAntIndexedDataObject):
	def __init__(self, df, definition):
		Builder1dDataObject.__init__(self, df)
		BuilderExpAntIndexedDataObject.__init__(self, df)
		DefinitionBuilder.build_from_definition(self, definition)

	def copy(self, name, category=None, label=None, description=None):
			return AntCharacteristics1dBuilder.build(
				df=self.df.copy(), name=name, category=category, label=label, description=description)


class AntCharacteristics1dBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build(df, name, category=None, label=None, description=None):
		definition = DefinitionBuilder().build1d(
			name=name, category=category, object_type='AntCharacteristics1d',
			label=label, description=description
		)
		return AntCharacteristics1d(df.sort_index(), definition)
