from DataManager.DataFileManager import DataFileManager
from DataObjects.Events2d import Events2dBuilder
from DataObjects.Filters import Filters
from DataObjects.TimeSeries2d import TimeSeries2dBuilder


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

	def to_2d(self, name1, name2, new_name, category, label, xlabel, ylabel, description):
		object_type1 = self.__dict__[name1].object_type
		object_type2 = self.__dict__[name2].object_type
		if object_type1 == object_type2 and self.__dict__[name1].array.index.equals(self.__dict__[name2].array.index):
			if object_type1 == 'Events':
				self.add_object(new_name, Events2dBuilder().build(
					event1=self.__dict__[name1], event2=self.__dict__[name2],
					name=new_name, category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description
				))
			elif object_type1 == 'TimeSeries':
				self.add_object(new_name, TimeSeries2dBuilder().build(
					ts1=self.__dict__[name1], ts2=self.__dict__[name2],
					name=new_name, category=category, label=label,
					xlabel=xlabel, ylabel=ylabel, description=description
				))
			else:
				raise TypeError(object_type1+' can not be gathered in 2d')

	def write(self, names):
		if isinstance(names, str):
			names = [names]
		for name in names:
			self.data_manager.write(self.__dict__[name])

	def copy(self, name, new_name, category, description, label, object_type):
		array = self.__dict__[name].copy(new_name, category, description, label, object_type)
		self.add_object(new_name, array)

	def filter(self, name1, name2, new_name, label, category, description):
		object_type1 = self.__dict__[name1].object_type
		object_type2 = self.__dict__[name2].object_type
		if object_type1 in ['Events', 'TimeSeries', 'Events2d', 'TimeSeries2d']\
			and object_type2 in ['Events', 'TimeSeries', 'Events2d', 'TimeSeries2d']:
			self.add_object(
				new_name,
				Filters().filter(self.__dict__[name1], self.__dict__[name2], new_name, label, category, description))
		else:
			raise TypeError('Filter can not be applied on '+object_type1+' or '+object_type2)

	def operation(self, name1, name2, fct, name_col=None):
		if self.__dict__[name1].object_type == 'TimeSeries':
			if self.__dict__[name2].object_type == 'Characteristics1d':
				self.__dict__[name1].operation_with_characteristics1d(self.__dict__[name2], fct)
			elif self.__dict__[name2].object_type == 'Characteristics2d':
				self.__dict__[name1].operation_with_characteristics2d(self.__dict__[name2], name_col, fct)
		else:
			raise IndexError(name1+' index names does not contains '+name2+' index names')
