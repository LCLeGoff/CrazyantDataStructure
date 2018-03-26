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

	def operation(self, name1, name2, fct, name_col=None):
		if self.__dict__[name1].object_type == 'TimeSeries':
			if self.__dict__[name2].object_type == 'Characteristics1d':
				self.__dict__[name1].operation_with_characteristics1d(self.__dict__[name2], fct)
			elif self.__dict__[name2].object_type == 'Characteristics2d':
				self.__dict__[name1].operation_with_characteristics2d(self.__dict__[name2], name_col, fct)
		else:
			raise IndexError(name1+' index names does not contains '+name2+' index names')
