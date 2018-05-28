import numpy as np

from DataManager.DataFileManager import DataFileManager
from DataObjectBuilders.Builder import Builder
from DataObjects.Events2d import Events2dBuilder
from DataObjects.Filters import Filters
from DataObjects.TimeSeries2d import TimeSeries2dBuilder
from PandasIndexManager.PandasIndexManager import PandasIndexManager
from Plotter.Plotter1d import Plotter1d
from Plotter.Plotter2d import Plotter2d


class ExperimentGroups:

	def __init__(self, root, group, id_exp_list):
		self.root = root
		self.group = group
		self.data_manager = DataFileManager(root, group)
		self.pandas_index_manager = PandasIndexManager()
		self.plotter2d = Plotter2d()
		self.id_exp_list = id_exp_list
		self.id_exp_list.sort()
		self.names = set()

	@staticmethod
	def turn_to_list(names):
		if isinstance(names, str):
			names = [names]
		return names

	def get_xname(self, name):
		return self.__dict__[name].df.columns[0]

	def get_yname(self, name):
		return self.__dict__[name].df.columns[1]

	def get_category(self, name):
		return self.__dict__[name].definition.category

	def get_label(self, name):
		return self.__dict__[name].definition.label

	def get_xlabel(self, name):
		return self.__dict__[name].definition.xlabel

	def get_ylabel(self, name):
		return self.__dict__[name].definition.ylabel

	def get_description(self, name):
		return self.__dict__[name].definition.description

	def set_id_exp_list(self, id_exp_list):
		if id_exp_list is None:
			id_exp_list = self.id_exp_list
		return id_exp_list

	def add_object(self, name, obj):
		self.__dict__[name] = obj
		self.names.add(name)

	def load(self, names):
		names = self.turn_to_list(names)
		for name in names:
			if name not in self.__dict__.keys():
				self.add_object(name, self.data_manager.load(name))

	def write(self, names):
		names = self.turn_to_list(names)
		for name in names:
			self.data_manager.write(self.__dict__[name])

	def delete_data(self, names):
		names = self.turn_to_list(names)
		for name in names:
			self.data_manager.delete(self.__dict__[name])

	def is_1d(self, name):
		object_type = self.get_object_type(name)
		if object_type in ['Events2d', 'TimeSeries2d', 'Characteristics2d']:
			return False
		elif object_type in ['Events1d', 'TimeSeries1d', 'Characteristics1d', 'AntCharacteristics1d']:
			return True
		else:
			raise TypeError('Object type '+object_type+' unknown')

	def rename_data(self, old_name, new_name):
		self.load(old_name)
		if self.is_1d(old_name):
			self.add_copy1d(name_to_copy=old_name, copy_name=new_name, copy_definition=True)
		else:
			self.add_copy2d(name_to_copy=old_name, copy_name=new_name, copy_definition=True)
		self.delete_data(old_name)
		self.write(new_name)

	def add_2d_from_1ds(
			self, name1, name2, result_name,
			xname=None, yname=None,
			category=None, label=None, xlabel=None, ylabel=None, description=None):

		object_type1 = self.get_object_type(name1)
		object_type2 = self.get_object_type(name2)

		if object_type1 == object_type2 and self.__dict__[name1].df.index.equals(self.__dict__[name2].df.index):
			if xname is None:
				xname = name1
			if yname is None:
				yname = name2
			if object_type1 == 'Events1d':
				self.add_object(result_name, Events2dBuilder().build_from_1d(
					event1=self.__dict__[name1], event2=self.__dict__[name2],
					name=result_name, xname=xname, yname=yname,
					category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description
				))
			elif object_type1 == 'TimeSeries1d':
				self.add_object(result_name, TimeSeries2dBuilder().build_from_1d(
					ts1=self.__dict__[name1], ts2=self.__dict__[name2],
					name=result_name, xname=xname, yname=yname,
					category=category, label=label,
					xlabel=xlabel, ylabel=ylabel, description=description
				))
			else:
				raise TypeError(object_type1+' can not be gathered in 2d')

	def get_object_type(self, name):
		return self.__dict__[name].object_type

	def plot_hist1d(self, name, bins, xscale=None, yscale=None, group=0):
		plot = Plotter1d()
		return plot.hist1d(self.__dict__[name], bins=bins, xscale=xscale, yscale=yscale, group=group)

	def add_copy1d(self, name_to_copy, copy_name, category=None, label=None, description=None, copy_definition=False):
		if copy_definition:
			obj = self.__dict__[name_to_copy].copy(
				name=copy_name,
				category=self.get_category(name_to_copy),
				label=self.get_label(name_to_copy),
				description=self.get_description(name_to_copy))
		else:
			obj = self.__dict__[name_to_copy].copy(name=copy_name, category=category, label=label, description=description)
		self.add_object(copy_name, obj)

	def add_copy2d(
			self, name_to_copy, copy_name, new_xname=None, new_yname=None,
			category=None, label=None, xlabel=None, ylabel=None, description=None,
			copy_definition=False):

		if copy_definition:
			obj = self.__dict__[name_to_copy].copy(
				name=copy_name,
				xname=self.get_xname(name_to_copy),
				yname=self.get_yname(name_to_copy),
				category=self.get_category(name_to_copy),
				label=self.get_label(name_to_copy),
				xlabel=self.get_xlabel(name_to_copy),
				ylabel=self.get_ylabel(name_to_copy),
				description=self.get_description(name_to_copy)
			)
		else:
			obj = self.__dict__[name_to_copy].copy(
				name=copy_name, xname=new_xname, yname=new_yname,
				category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description)
		self.add_object(copy_name, obj)

	def add_new1d_empty(self, name, object_type, category=None, label=None, description=None):
		df = self.__create_empty_1d_df(name, object_type)
		obj = Builder.build1d(
			df=df, name=name, object_type=object_type, category=category, label=label, description=description)
		self.add_object(name, obj)

	def __create_empty_1d_df(self, name, object_type):
		if object_type in ['Events1d', 'TimeSeries1d']:
			df = self.pandas_index_manager.create_empty_exp_ant_frame_indexed_1d_df(name)
		elif object_type in ['AntCharacteristics1d']:
			df = self.pandas_index_manager.create_empty_exp_ant_indexed_df(name)
		elif object_type in ['Characteristics1d']:
			df = self.pandas_index_manager.create_empty_exp_indexed_df(name)
		elif object_type in ['Events2d', 'TimeSeries2d', 'Characteristics2d']:
			raise TypeError('Object in 2d')
		else:
			raise IndexError('Object type ' + object_type + ' unknown')
		return df

	def add_new2d_empty(
			self, name, xname, yname, object_type, category=None,
			label=None, xlabel=None, ylabel=None, description=None):

		obj = Builder.build2d_from_array(
			array=np.zeros((0, 5)), name=name, xname=xname, yname=yname,
			object_type=object_type, category=category,
			label=label, xlabel=xlabel, ylabel=ylabel, description=description)

		self.add_object(name, obj)

	def add_new1d_from_df(self, df, name, object_type, category=None, label=None, description=None):
		obj = Builder.build1d(
			df=df, name=name, object_type=object_type, category=category, label=label, description=description)
		self.add_object(name, obj)

	def add_new2d_from_df(
			self, df, name, xname, yname, object_type, category=None,
			label=None, xlabel=None, ylabel=None, description=None):

		obj = Builder.build2d_from_df(
			df=df, name=name, xname=xname, yname=yname,
			object_type=object_type, category=category,
			label=label, xlabel=xlabel, ylabel=ylabel, description=description)

		self.add_object(name, obj)

	def add_new1d_from_array(self, array, name, object_type, category=None, label=None, description=None):

		df = self.__convert_array_to_1d_df(array, name, object_type)

		obj = Builder.build1d(
			df=df, name=name, object_type=object_type, category=category, label=label, description=description)

		self.add_object(name, obj)

	def __convert_array_to_1d_df(self, array, name, object_type):
		if object_type in ['Events1d', 'TimeSeries1d']:
			df = self.pandas_index_manager.convert_to_exp_ant_frame_indexed_1d_df(array, name)
		elif object_type in ['AntCharacteristics1d']:
			df = self.pandas_index_manager.convert_to_exp_ant_indexed_df(array, name)
		elif object_type in ['Characteristics1d']:
			df = self.pandas_index_manager.convert_to_exp_indexed_df(array, name)
		elif object_type in ['Events2d', 'TimeSeries2d', 'Characteristics2d']:
			raise TypeError('Object in 2d')
		else:
			raise IndexError('Object type ' + object_type + ' unknown')
		return df

	def add_from_filtering(self, name_to_filter, name_filter, result_name, label=None, category=None, description=None):
		object_type1 = self.get_object_type(name_to_filter)
		object_type2 = self.get_object_type(name_filter)
		if object_type1 in ['Events1d', 'TimeSeries1d', 'Events2d', 'TimeSeries2d']\
			and object_type2 in ['Events1d', 'TimeSeries1d', 'Events2d', 'TimeSeries2d']:
			self.add_object(
				result_name,
				Filters().filter(
					obj=self.__dict__[name_to_filter], event=self.__dict__[name_filter],
					name=result_name, label=label, category=category, description=description))
		else:
			raise TypeError('Filter can not be applied on '+object_type1+' or '+object_type2)

	def operation(self, name, fct):
		self.__dict__[name].operation(fct)

	def operation_between_2names(self, name1, name2, fct, name_col=None):
		if self.get_object_type(name1) == 'TimeSeries1d':
			if self.get_object_type(name2) == 'TimeSeries1d':
				self.__dict__[name1].operation_with_data1d(self.__dict__[name2], fct)
			elif self.get_object_type(name2) == 'Characteristics1d':
				self.__dict__[name1].operation_with_data1d(self.__dict__[name2], fct)
			elif self.get_object_type(name2) == 'Characteristics2d':
				self.__dict__[name1].operation_with_data2d(self.__dict__[name2], name_col, fct)
			else:
				raise TypeError('Operation not defined between TimeSeries and ' + self.get_object_type(name2))
		elif self.get_object_type(name1) == 'Characteristics1d':
			if self.get_object_type(name2) == 'Characteristics1d':
				self.__dict__[name1].operation_with_data1d(self.__dict__[name2], fct)
			else:
				raise TypeError('Operation not defined between Characteristics1d and ' + self.get_object_type(name2))
		else:
			raise TypeError(
				'Operation not defined between ' + self.get_object_type(name2) + ' and ' + self.get_object_type(name2))

	def add_event_extracted_from_timeseries(self, name_ts, name_extracted_events, label=None, category=None, description=None):
		if self.get_object_type(name_ts) == 'TimeSeries1d':
			event = self.__dict__[name_ts].extract_event(name=name_extracted_events, category=category, label=label, description=description)
			self.add_object(name_extracted_events, event)
