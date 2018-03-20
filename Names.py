class Names:
	def __init__(self, name, attr_dict):
		self.name = name
		self.dict = attr_dict
		for key, value in attr_dict.items():
			self.__dict__[key] = value
