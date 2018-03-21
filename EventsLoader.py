import pandas as pd

from Events import Events


class EventsLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'

	def load(self, name):
		add = self.root+name.category+'/'+name.name+'.csv'
		return Events(pd.read_csv(add, index_col=['id_exp', 'id_ant', 'frame']), name)