from DataManager.Deleters.AntCharacteristicsDeleter import AntCharacteristics1dDeleter
from DataManager.Deleters.DefinitionDeleter import DefinitionDeleter
from DataManager.Deleters.EventsDeleter import Events1dDeleter, Events2dDeleter
from DataManager.Deleters.TimeSeriesDeleter import TimeSeriesDeleter


class DataDeleter:
	def __init__(self, root, group):
		self.definition_deleter = DefinitionDeleter(root, group)
		self.time_series_deleter = TimeSeriesDeleter(root, group)
		self.events1d_deleter = Events1dDeleter(root, group)
		self.events2d_deleter = Events2dDeleter(root, group)
		self.ant_characteristics1d_deleter = AntCharacteristics1dDeleter(root, group)

	def delete(self, obj):
		if obj.object_type == 'TimeSeries1d':
			self.time_series_deleter.delete(obj)
		elif obj.object_type == 'Events1d':
			self.events1d_deleter.delete(obj)
		elif obj.object_type == 'Events2d':
			self.events2d_deleter.write(obj)
		elif obj.object_type == 'AntCharacteristics1d':
			self.ant_characteristics1d_deleter.delete(obj)
		else:
			raise ValueError(obj.name+' has no defined object type')
		self.definition_deleter.delete(obj.definition)
