import pandas as pd

from DataObjects.Events1d import Events1d


class EventsLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'

	def load(self, definition):
		add = self.root + definition.category + '/' + definition.name + '.csv'
		return Events1d(pd.read_csv(add, index_col=['id_exp', 'id_ant', 'frame']), definition)
