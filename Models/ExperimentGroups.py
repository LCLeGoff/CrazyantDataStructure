from DataManager.DataFileManager import DataFileManager


class ExperimentGroups:

	def __init__(self, root, group, id_exp_list):
		self.root = root
		self.group = group
		self.data_manager = DataFileManager(root, group)
		self.id_exp_list = id_exp_list
		self.names = set()

	def add_object(self, name, obj):
		self.__dict__[name] = obj
		self.names.add(name)

	def load(self, names):
		if isinstance(names, str):
			names = [names]
		for name in names:
			if name not in self.__dict__.keys():
				self.add_object(name, self.data_manager.load(name))

	def write(self, names):
		if isinstance(names, str):
			names = [names]
		for name in names:
			self.data_manager.write(self.__dict__[name])

	def copy(self, name, new_name, category, description, label, object_type):
		array = self.__dict__[name].copy(new_name, category, description, label, object_type)
		self.add_object(new_name, array)

	def operation(self, name1, name2, fct, name_col=None):
		if self.__dict__[name1].object_type == 'TimeSeries':
			if self.__dict__[name2].object_type == 'Characteristics1d':
				self.__dict__[name1].operation_with_characteristics1d(self.__dict__[name2], fct)
			elif self.__dict__[name2].object_type == 'Characteristics2d':
				self.__dict__[name1].operation_with_characteristics2d(self.__dict__[name2], name_col, fct)
		else:
			raise IndexError(name1+' index names does not contains '+name2+' index names')
