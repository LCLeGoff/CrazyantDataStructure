import pandas as pd


class TimeSeriesWriter:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.categories = dict()

	def write(self, ts_list):
		if not(isinstance(ts_list, list)):
			ts_list = [ts_list]
		ts_dict = dict()
		for ts in ts_list:
			if not(ts.category in ts_dict.keys()):
				ts_dict[ts.category] = []
			ts_dict[ts.category].append(ts)
		if 'Raw' in ts_dict.keys():
			raise OSError('not allowed to modify TimeSeries of the category Raw')
		else:
			# TODO: useless for now, had to fix this
			for category in ts_dict.keys():
				add = self.root + category + '/TimeSeries.csv'
				array = pd.read_csv(add, index_col=['id_exp', 'id_ant', 'frame'])
				for ts in ts_dict[category]:
					array[ts.name] = ts.array
				array.to_csv(add)
