import pandas as pd

from DataObjects.TimeSeries1d import TimeSeries1d


class TimeSeriesLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.categories = dict()

	def load_category(self, category):
		add = self.root + category + '/TimeSeries.csv'
		if not(category in self.categories.keys()):
			self.categories[category] = pd.read_csv(add, index_col=['id_exp', 'id_ant', 'frame'])

	def load(self, definition):
		self.load_category(definition.category)
		return TimeSeries1d(pd.DataFrame(self.categories[definition.category][definition.name]), definition)
