from Loaders.CharacteristicsLoader import Characteristics1dLoader, Characteristics2dLoader
from Loaders.EventsLoader import EventsLoader
from Builders.NameBuilder import NameBuilder
from Loaders.TimeSeriesLoader import TimeSeriesLoader


class DataLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.name_builder = NameBuilder(root, group)
		self.time_series_loader = TimeSeriesLoader(root, group)
		self.events_loader = EventsLoader(root, group)
		self.characteristics1d_builder = Characteristics1dLoader(root, group)
		self.characteristics2d_builder = Characteristics2dLoader(root, group)

	def load(self, name):
		name_class = self.name_builder.build(name)
		if name_class.object_type == 'TimeSeries':
			res = self.time_series_loader.load(name_class)
		elif name_class.object_type == 'Events':
			res = self.events_loader.load(name_class)
		elif name_class.object_type == 'Characteristics1d':
			res = self.characteristics1d_builder.load(name_class)
		elif name_class.object_type == 'Characteristics2d':
			res = self.characteristics2d_builder.load(name_class)
		else:
			raise ValueError(name+' has no defined object type')
		return res
