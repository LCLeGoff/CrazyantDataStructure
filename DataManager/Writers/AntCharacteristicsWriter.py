
class AntCharacteristics1dWriter:
	def __init__(self, root, group):
		self.root = root+group+'/'

	def write(self, ant_chara):
		if ant_chara.category == 'Raw':
			raise OSError('not allowed to modify AntCharacteristics of the category Raw')
		else:
			add = self.root + ant_chara.category + '/' + ant_chara.name + '.csv'
			ant_chara.df.to_csv(add)
