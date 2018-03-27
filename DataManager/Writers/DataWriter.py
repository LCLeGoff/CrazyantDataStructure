from DataManager.Writers.DefinitionWriter import DefinitionWriter
from DataManager.Writers.TimeSeriesWriter import TimeSeriesWriter


class DataWriter:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.definition_writer = DefinitionWriter(root, group)
		self.time_series_writer = TimeSeriesWriter(root, group)
		# self.events_writer = EventsWriter(root, group)
		# self.characteristics1d_writer = Characteristics1dWriter(root, group)
		# self.characteristics2d_writer = Characteristics2dWriter(root, group)

	def write(self, obj):
		if obj.object_type == 'TimeSeries':
			self.time_series_writer.write(obj)
			self.definition_writer.write(obj.definition)

		# elif definition.object_type == 'Events':
		# 	self.events_writer.load(definition)
		# elif definition.object_type == 'Characteristics1d':
		# 	self.characteristics1d_writer.load(definition)
		# elif definition.object_type == 'Characteristics2d':
		# 	self.characteristics2d_writer.load(definition)
		else:
			raise ValueError(obj.name+' has no defined object type')
