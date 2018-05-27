from DataManager.Writers.AntCharacteristicsWriter import AntCharacteristics1dWriter
from DataManager.Writers.DefinitionWriter import DefinitionWriter
from DataManager.Writers.EventsWriter import Events1dWriter, Events2dWriter
from DataManager.Writers.TimeSeriesWriter import TimeSeriesWriter


class DataWriter:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.definition_writer = DefinitionWriter(root, group)
		self.time_series_writer = TimeSeriesWriter(root, group)
		self.events_writer = Events1dWriter(root, group)
		self.events2d_writer = Events2dWriter(root, group)
		self.ant_characteristics1d_writer = AntCharacteristics1dWriter(root, group)

	def write(self, obj):
		if obj.object_type == 'TimeSeries':
			self.time_series_writer.write(obj)
		elif obj.object_type == 'Events':
			self.events_writer.write(obj)
		elif obj.object_type == 'Events2d':
			self.events2d_writer.write(obj)
		elif obj.object_type == 'AntCharacteristics1d':
			self.ant_characteristics1d_writer.write(obj)
		else:
			raise ValueError(obj.name+' has no defined object type')
		self.definition_writer.write(obj.definition)
