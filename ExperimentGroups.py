from DataLoader import DataLoader


class ExperimentGroups:

	def __init__(self, root, group):
		self.root = root
		self.group = group
		self.data_loader = DataLoader(root, group)

	def load(self, name):
		self.__dict__[name] = self.data_loader.load(name)
