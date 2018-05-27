import pandas as pd

from DataObjects.Definitions import DefinitionBuilder
from DataObjectBuilders.Builder2dDataObject import Builder2dDataObject
from DataObjectBuilders.BuilderExpAntFrameIndexedDataObject import BuilderExpAntFrameIndexedDataObject


class Events2d(Builder2dDataObject, BuilderExpAntFrameIndexedDataObject):
	def __init__(self, df, definition):
		Builder2dDataObject.__init__(self, df)
		BuilderExpAntFrameIndexedDataObject.__init__(self, df)
		DefinitionBuilder.build_from_definition(self, definition)

	def copy(self, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
			return Events2dBuilder.build_from_df(
				self.df.copy(), name=name, xname=xname, yname=yname,
				category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description)


class Events2dBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build_from_1d(
			event1, event2, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
		df = pd.DataFrame(index=event1.df.index)
		df[xname] = event1.df
		df[yname] = event2.df
		definition = DefinitionBuilder().build2d(
			name=name, category=category, object_type='Events2d',
			label=label, xlabel=xlabel, ylabel=ylabel, description=description
		)
		return Events2d(df, definition)

	@staticmethod
	def build_from_df(df, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
		df.columns = [xname, yname]
		definition = DefinitionBuilder().build2d(
			name=name, category=category, object_type='Events2d',
			label=label, xlabel=xlabel, ylabel=ylabel, description=description
		)
		return Events2d(df.sort_index(), definition)
