import numpy as np
import pandas as pd

from DataManager.DataFileManager import DataFileManager
from DataObjectBuilders.Builder import Builder
from DataObjects.Events2d import Events2dBuilder
from DataObjects.Filters import Filters
from DataObjects.TimeSeries2d import TimeSeries2dBuilder
from Plotter.Plotter1d import Plotter1d
from Plotter.Plotter2d import Plotter2d


class ExperimentGroups:

	def __init__(self, root, group, id_exp_list):
		self.root = root
		self.group = group
		self.data_manager = DataFileManager(root, group)
		self.plotter2d = Plotter2d()
		self.id_exp_list = id_exp_list
		self.id_exp_list.sort()
		self.names = set()

	def set_id_exp_list(self, id_exp_list):
		if id_exp_list is None:
			id_exp_list = self.exp.id_exp_list
		return id_exp_list

	def add_object(self, name, obj):
		self.__dict__[name] = obj
		self.names.add(name)

	def load(self, names):
		if isinstance(names, str):
			names = [names]
		for name in names:
			if name not in self.__dict__.keys():
				self.add_object(name, self.data_manager.load(name))

	def to_2d(
			self, name1, name2, new_name,
			new_name1=None, new_name2=None,
			category=None, label=None, xlabel=None, ylabel=None, description=None):
		object_type1 = self.__dict__[name1].object_type
		object_type2 = self.__dict__[name2].object_type
		if object_type1 == object_type2 and self.__dict__[name1].array.index.equals(self.__dict__[name2].array.index):
			if new_name1 is None:
				new_name1 = name1
			if new_name2 is None:
				new_name2 = name2
			if object_type1 == 'Events':
				self.add_object(new_name, Events2dBuilder().build_from_1d(
					event1=self.__dict__[name1], event2=self.__dict__[name2],
					name=new_name, xname=new_name1, yname=new_name2,
					category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description
				))
			elif object_type1 == 'TimeSeries':
				self.add_object(new_name, TimeSeries2dBuilder().build_from_1d(
					ts1=self.__dict__[name1], ts2=self.__dict__[name2],
					name=new_name, xname=new_name1, yname=new_name2,
					category=category, label=label,
					xlabel=xlabel, ylabel=ylabel, description=description
				))
			else:
				raise TypeError(object_type1+' can not be gathered in 2d')

	def plot_repartition(self, name):
		plot = Plotter2d()
		return plot.repartition(self.__dict__[name], title_prefix=self.group)

	def plot_repartition_hist(self, name):
		plot = Plotter2d()
		return plot.hist2d(self.__dict__[name], bins=[75, 50], title_prefix=self.group)

	def plot_hist1d(self, name, bins, xscale=None, yscale=None, group=0):
		plot = Plotter1d()
		return plot.hist1d(self.__dict__[name], bins=bins, xscale=xscale, yscale=yscale, group=group)

	def write(self, names):
		if isinstance(names, str):
			names = [names]
		for name in names:
			self.data_manager.write(self.__dict__[name])

	def copy1d(self, name, new_name, category=None, label=None, description=None):
		obj = self.__dict__[name].copy(name=new_name, category=category, label=label, description=description)
		self.add_object(new_name, obj)

	def copy2d(
			self, name, new_name, new_xname, new_yname,
			category=None, label=None, xlabel=None, ylabel=None, description=None):
		array = self.__dict__[name].copy(
			name=new_name, xname=new_xname, yname=new_yname,
			category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description)
		self.add_object(new_name, array)

	def build1d_empty(self, name, object_type, category=None, label=None, description=None):
		obj = Builder.build1d(
			array=pd.DataFrame(
				columns=['id_exp', 'id_ant', 'frame', name]).set_index(['id_exp', 'id_ant', 'frame']),
			name=name, object_type=object_type, category=category, label=label, description=description)
		self.add_object(name, obj)

	def build2d_empty(self, name, object_type, category=None, label=None, description=None):
		obj = Builder.build1d(
			array=np.zeros((0, 5)), name=name, object_type=object_type, category=category, label=label, description=description)
		self.add_object(name, obj)

	def build1d(self, array, name, object_type, category=None, label=None, description=None):
		obj = Builder.build1d(
			array=array, name=name, object_type=object_type, category=category, label=label, description=description)
		self.add_object(name, obj)

	def build2d_from_array(
			self, array, name, xname, yname, object_type, category=None,
			label=None, xlabel=None, ylabel=None, description=None):
		obj = Builder.build2d_from_array(
			array=array, name=name, xname=xname, yname=yname,
			object_type=object_type, category=category,
			label=label, xlabel=xlabel, ylabel=ylabel, description=description)
		self.add_object(name, obj)

	def filter(self, name1, name2, new_name, label=None, category=None, description=None):
		object_type1 = self.__dict__[name1].object_type
		object_type2 = self.__dict__[name2].object_type
		if object_type1 in ['Events', 'TimeSeries', 'Events2d', 'TimeSeries2d']\
			and object_type2 in ['Events', 'TimeSeries', 'Events2d', 'TimeSeries2d']:
			self.add_object(
				new_name,
				Filters().filter(
					obj=self.__dict__[name1], event=self.__dict__[name2],
					name=new_name, label=label, category=category, description=description))
		else:
			raise TypeError('Filter can not be applied on '+object_type1+' or '+object_type2)

	def operation(self, name1, name2, fct, name_col=None):
		if self.__dict__[name1].object_type == 'TimeSeries':
			if self.__dict__[name2].object_type == 'TimeSeries':
				self.__dict__[name1].operation_with_data1d(self.__dict__[name2], fct)
			elif self.__dict__[name2].object_type == 'Characteristics1d':
				self.__dict__[name1].operation_with_data1d(self.__dict__[name2], fct)
			elif self.__dict__[name2].object_type == 'Characteristics2d':
				self.__dict__[name1].operation_with_data2d(self.__dict__[name2], name_col, fct)
			else:
				raise TypeError('Operation not defined between TimeSeries and '+self.__dict__[name2].object_type)
		elif self.__dict__[name1].object_type == 'Characteristics1d':
			if self.__dict__[name2].object_type == 'Characteristics1d':
				self.__dict__[name1].operation_with_data1d(self.__dict__[name2], fct)
			else:
				raise TypeError('Operation not defined between Characteristics1d and '+self.__dict__[name2].object_type)
		else:
			raise TypeError(
				'Operation not defined between '+self.__dict__[name2].object_type+' and '+self.__dict__[name2].object_type)

	def extract_event(self, name, new_name, label=None, category=None, description=None):
		if self.__dict__[name].object_type == 'TimeSeries':
			event = self.__dict__[name].extract_event(name=new_name, category=category, label=label, description=description)
			self.add_object(new_name, event)
