
class Events1dWriter:
	def __init__(self, root, group):
		self.root = root+group+'/'

	def write(self, event):
		if event.category == 'Raw':
			raise OSError('not allowed to modify Events of the category Raw')
		else:
			add = self.root + event.category + '/'+event.name+'.csv'
			event.df.to_csv(add)


class Events2dWriter:
	def __init__(self, root, group):
		self.root = root+group+'/'

	def write(self, event2d):
		if event2d.category == 'Raw':
			raise OSError('not allowed to modify Events2d of the category Raw')
		else:
			add = self.root + event2d.category + '/'+event2d.name+'.csv'
			event2d.df.to_csv(add)