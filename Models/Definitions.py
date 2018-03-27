class Definitions:
	def __init__(self, name, attr_dict):
		self.name = name
		self.dict = attr_dict
		for key, value in attr_dict.items():
			self.__dict__[key] = value

	def copy(self, new_name, new_category, new_description, new_label, new_object_type):
		def_dict = self.dict
		def_dict['category'] = new_category
		def_dict['description'] = new_description
		def_dict['label'] = new_label
		def_dict['object_type'] = new_object_type
		return Definitions(new_name, def_dict)
