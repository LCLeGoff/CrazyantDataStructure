from DataObjects.Definitions import DefinitionBuilder
from Builders.DefinitionDataObjectBuilder import DefinitionDataObjectBuilder
from DataObjectBuilders.BuilderExpAntFrameIndexed1dDataObject import BuilderExpAntFrameIndexed1dDataObject


class TimeSeries(BuilderExpAntFrameIndexed1dDataObject):
	def __init__(self, array, definition):
		# TODO: Fix this conception mistake
		array.columns = [definition.name]
		BuilderExpAntFrameIndexed1dDataObject.__init__(self, array)
		DefinitionDataObjectBuilder.build(self, definition)

	def copy(self, name, category, label, description):
			return TimeSeriesBuilder.build(self.array, name, category, label, description)

	def operation_with_characteristics1d(self, chara, fct):
		self.array[self.name_col] = fct(self.array[self.name_col], chara.array[chara.name_col])

	def operation_with_characteristics2d(self, chara, name_col, fct):
		self.array[self.name_col] = fct(self.array[self.name_col], chara.array[name_col])


class TimeSeriesBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build(array, name, category, label, description):
		definition = DefinitionBuilder().build1d(
			name=name, category=category, object_type='TimeSeries',
			label=label, description=description
		)
		return TimeSeries(array.copy(), definition)