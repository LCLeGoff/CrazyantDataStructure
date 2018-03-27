class DefinitionDataObjectBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build(class_self, definition):
		class_self.name = definition.name
		class_self.definition = definition
		for key, value in definition.dict.items():
			class_self.__dict__[key] = value
