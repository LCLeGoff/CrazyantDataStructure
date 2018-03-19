from JsonFiles import JsonFiles as js
from Names import Names


class NameBuilder:
	def __init__(self, root, group):
		self.root = root
		self.group = group
		self.reference_dict = js.import_obj(self.root+'/'+self.group+'/ref_names.csv')

	def build(self, name):
		return Names(self.reference_dict[name])
