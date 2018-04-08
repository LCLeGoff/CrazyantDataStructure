
class EventsWriter:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.categories = dict()

	def write(self, event):
		if event.category == 'Raw':
			raise OSError('not allowed to modify Events of the category Raw')
		else:
			add = self.root + event.category + '/'+event.name+'.csv'
			event.array.to_csv(add)
