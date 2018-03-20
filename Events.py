class Events:
	def __init__(self, tab, name):
		self.name = name
		for key, value in name.dict.items():
			self.__dict__[key] = value
		self.tab = tab
