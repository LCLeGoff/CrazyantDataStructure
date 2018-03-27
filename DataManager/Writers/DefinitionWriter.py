from Tools.JsonFiles import import_obj, write_obj


class DefinitionWriter:
	def __init__(self, root, group):
		self.add = root+group+'/definition_dict.json'

	def write(self, definition):
		def_dict = import_obj(self.add)
		def_dict[definition.name] = definition.dict
		write_obj(self.add, def_dict)
