from Loaders.DataLoader import DataLoader


class ExperimentGroups:

	def __init__(self, root, group, id_exp_list):
		self.root = root
		self.group = group
		self.data_loader = DataLoader(root, group)
		self.id_exp_list = id_exp_list

	def load(self, names):
		if isinstance(names, list):
			for name in names:
				if name not in self.__dict__.keys():
					self.__dict__[name] = self.data_loader.load(name)
		else:
			self.__dict__[names] = self.data_loader.load(names)
