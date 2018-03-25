class CharacteristicObjectBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build(class_self, name):
		class_self.name = name.name
		for key, value in name.dict.items():
			class_self.__dict__[key] = value
