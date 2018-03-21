from CharacteristicsLoader import CharacteristicsLoader
from EventsLoader import EventsLoader
from NameBuilder import NameBuilder
from TimeSeriesLoader import TimeSeriesLoader


class DataLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.name_builder = NameBuilder(root, group)
		self.time_series_loader = TimeSeriesLoader(root, group)
		self.events_loader = EventsLoader(root, group)
		self.characteristics_builder = CharacteristicsLoader(root, group)

	def load(self, name):
		name_class = self.name_builder.build(name)
		if name_class.object_type == 'TimeSeries':
			res = self.time_series_loader.load(name_class)
		elif name_class.object_type == 'Events':
			res = self.events_loader.load(name_class)
		elif name_class.object_type == 'Characteristics':
			res = self.characteristics_builder.build(name_class)
		else:
			raise ValueError(name+' has no defined object type')
		return res
