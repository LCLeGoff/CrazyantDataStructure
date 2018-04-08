import pandas as pd


class TimeSeriesWriter:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.categories = dict()

	def write(self, ts):
		if ts.category == 'Raw':
			raise OSError('not allowed to modify TimeSeries of the category Raw')
		else:
			add = self.root + ts.category + '/TimeSeries.csv'
			array = pd.read_csv(add, index_col=['id_exp', 'id_ant', 'frame'])
			array[ts.name] = ts.array
			array.to_csv(add)
