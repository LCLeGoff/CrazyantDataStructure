from DataManager.Loaders.CharacteristicsLoader import Characteristics1dLoader, Characteristics2dLoader
from DataManager.Loaders.Data2dLoader import Data2dLoader
from DataManager.Loaders.EventsLoader import EventsLoader
from DataManager.Loaders.DefinitionLoader import DefinitionLoader
from DataManager.Loaders.TimeSeriesLoader import TimeSeriesLoader


class DataLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.definition_loader = DefinitionLoader(root, group)
		self.time_series_loader = TimeSeriesLoader(root, group)
		self.time_series_loader.load_category('Raw')
		self.events_loader = EventsLoader(root, group)
		self.data2d_loader = Data2dLoader(root, group)
		self.characteristics1d_loader = Characteristics1dLoader(root, group)
		self.characteristics2d_loader = Characteristics2dLoader(root, group)

	def load(self, name):
		definition = self.definition_loader.build(name)
		if definition.object_type == 'TimeSeries':
			res = self.time_series_loader.load(definition)
		elif definition.object_type == 'Events':
			res = self.events_loader.load(definition)
		elif definition.object_type == 'Data2d':
			res = self.data2d_loader.load(definition)
		elif definition.object_type == 'Characteristics1d':
			res = self.characteristics1d_loader.load(definition)
		elif definition.object_type == 'Characteristics2d':
			res = self.characteristics2d_loader.load(definition)
		else:
			raise ValueError(name+' has no defined object type')
		return res
