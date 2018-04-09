
class Events2dWriter:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.categories = dict()

	def write(self, event2d):
		if event2d.category == 'Raw':
			raise OSError('not allowed to modify Events2d of the category Raw')
		else:
			add = self.root + event2d.category + '/'+event2d.name+'.csv'
			event2d.array.to_csv(add)