from NameBuilder import NameBuilder

class ExperimentGroups:

	def __init__(self, root, group):
		self.root = root
		self.group = group
		self.name_builder = NameBuilder(self.root, self.group)

	def get(self, name):
		name_class = self.name_builder.build(name)
		object = name_class.category
		category = name_class.category

		self.__dict__[name] =
