import pandas as pd

from NameBuilder import NameBuilder
from TimeSeriesBuilder import TimeSeriesBuilder


class DataLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.name_builder = NameBuilder(root, group)
		self.time_series_loader = TimeSeriesLoader(root, group)
		self.list_loaded_names = []

	def load(self, name):
		if name in self.list_loaded_names:
			print(name+' already loaded')
		else:
			name_class = self.name_builder.build(name)
			object_type = name_class.object_type
			category = name_class.category
			if object_type == 'TimeSeries':
				res = self.time_series_loader.load(category, name)
			else:
				res = None
			self.list_loaded_names.append(name)
			return res


class TimeSeriesLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'

	def load(self, category, name):
		add = self.root+category+'/TimeSeries.csv'
		return TimeSeriesBuilder.build(pd.read_csv(add).pivot_table(name, ['id_exp', 'id_ant', 'frame']))
