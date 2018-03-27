import pandas as pd

from Models.Events import Events


class EventsLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'

	def load(self, definition):
		add = self.root + definition.category + '/' + definition.name + '.csv'
		return Events(pd.read_csv(add, index_col=['id_exp', 'id_ant', 'frame']), definition)
