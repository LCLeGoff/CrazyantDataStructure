from JsonFiles import JsonFiles as js
from Names import Names


class NameBuilder:
	def __init__(self, root, group):
		self.reference_dict = js.import_obj(root+group+'/ref_names.json')

	def build(self, name):
		return Names(self.reference_dict[name])
