import pandas as pd

from Models.TimeSeries import TimeSeries


class TimeSeriesLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.categories = dict()

	def load(self, name):
		add = self.root+name.category+'/TimeSeries.csv'
		if not(name.category in self.categories.keys()):
			self.categories[name.category] = pd.read_csv(add, index_col=['id_exp', 'id_ant', 'frame'])
		return TimeSeries(pd.DataFrame(self.categories[name.category][name.name]), name)
