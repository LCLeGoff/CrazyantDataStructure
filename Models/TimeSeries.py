from Builders.DefinitionObjectBuilder import DefinitionObjectBuilder
from IndexedSeries.ExpAntFrameIndexedSeries import ExpAntFrameIndexedSeries


class TimeSeries(ExpAntFrameIndexedSeries):
	def __init__(self, array, definition):
		# TODO: Fix this conception mistake
		array.columns = [definition.name]
		ExpAntFrameIndexedSeries.__init__(self, array)
		DefinitionObjectBuilder.build(self, definition)

	def copy(self, new_name, new_category, new_description, new_label, new_object_type):
			return TimeSeries(
				self.array.copy(),
				self.definition.copy(new_name, new_category, new_description, new_label, new_object_type))

	def operation_with_characteristics1d(self, chara, fct):
		self.array[self.name_col] = fct(self.array[self.name_col], chara.array[chara.name_col])

	def operation_with_characteristics2d(self, chara, name_col, fct):
		self.array[self.name_col] = fct(self.array[self.name_col], chara.array[name_col])

