import pandas as pd

from EventsBuilder import EventsBuilder
from NameBuilder import NameBuilder
from TimeSeriesBuilder import TimeSeriesBuilder


class DataLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.name_builder = NameBuilder(root, group)
		self.time_series_loader = TimeSeriesLoader(root, group)
		self.events_loader = EventsLoader(root, group)

	def load(self, name):
		name_class = self.name_builder.build(name)
		object_type = name_class.object_type
		category = name_class.category
		if object_type == 'TimeSeries':
			res = self.time_series_loader.load(category, name)
		elif object_type == 'Events':
			res = self.events_loader.load(category, name)
		else:
			res = None
		return res


class TimeSeriesLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.categories = dict()

	def load(self, category, name):
		add = self.root+category+'/TimeSeries.csv'
		if not(category in self.categories.keys()):
			self.categories[category] = pd.read_csv(add, index_col=['id_exp', 'id_ant', 'frame'])
		return TimeSeriesBuilder.build(self.categories[category][name])


class EventsLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'

	def load(self, category, name):
		add = self.root+category+'/'+name+'.csv'
		return EventsBuilder.build(pd.read_csv(add, index_col=['id_exp', 'id_ant', 'frame']))
