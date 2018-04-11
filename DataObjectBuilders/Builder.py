from DataObjects.Characteristics1d import Characteristics1dBuilder
from DataObjects.Characteristics2d import Characteristics2dBuilder
from DataObjects.Definitions import DefinitionBuilder
from DataObjects.Events1d import Events1dBuilder
from DataObjects.Events2d import Events2dBuilder
from DataObjects.TimeSeries1d import TimeSeries1dBuilder
from DataObjects.TimeSeries2d import TimeSeries2dBuilder


class Builder:
	def __init__(self):
		pass

	@staticmethod
	def build1d(array, name, object_type, category=None, label=None, description=None):
		if object_type == 'TimeSeries1d':
			return TimeSeries1dBuilder.build(
				array=array.copy(), name=name, category=category, label=label, description=description)
		elif object_type == 'Events1d':
			return Events1dBuilder.build(
				array=array.copy(), name=name, category=category, label=label, description=description)
		elif object_type == 'Characteristics1d':
			return Characteristics1dBuilder.build(
				array=array.copy(), name=name, category=category, label=label, description=description)
		else:
			raise TypeError('Type '+object_type+' is unknown or 2d')

	@staticmethod
	def build2d_from_array(
			array, name, xname, yname, object_type, category=None, label=None, xlabel=None, ylabel=None, description=None):
		if object_type == 'TimeSeries1d':
			return TimeSeries2dBuilder.build_from_array(
				array=array.copy(), name=name, xname=xname, yname=yname,
				category=category, label=label, xlabel=xlabel, ylabel=ylabel,
				description=description)
		elif object_type == 'Events1d':
			return Events2dBuilder.build_from_array(
				array=array.copy(), name=name, xname=xname, yname=yname,
				category=category, label=label, xlabel=xlabel, ylabel=ylabel,
				description=description)
		elif object_type == 'Characteristics1d':
			return Characteristics2dBuilder.build_from_array(
				array=array.copy(), name=name, xname=xname, yname=yname,
				category=category, label=label, xlabel=xlabel, ylabel=ylabel,
				description=description)
		else:
			raise TypeError('Type '+object_type+' is unknown or 1d')
