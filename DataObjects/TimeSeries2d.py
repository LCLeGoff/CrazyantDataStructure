import pandas as pd

from DataObjects.Definitions import DefinitionBuilder
from DataObjectBuilders.Builder2dDataObject import Builder2dDataObject
from DataObjectBuilders.BuilderExpAntFrameIndexedDataObject import BuilderExpAntFrameIndexedDataObject


class TimeSeries2d(Builder2dDataObject, BuilderExpAntFrameIndexedDataObject):
	def __init__(self, df, definition):
		Builder2dDataObject.__init__(self, df)
		BuilderExpAntFrameIndexedDataObject.__init__(self, df)
		DefinitionBuilder.build_from_definition(self, definition)


class TimeSeries2dBuilder:
	def __init__(self):
		pass

	@staticmethod
	def build_from_1d(ts1, ts2, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
		df = pd.DataFrame(index=ts1.df.index)
		df[xname] = ts1.df
		df[yname] = ts2.df
		definition = DefinitionBuilder().build2d(
			name=name, category=category, object_type='TimeSeries2d',
			label=label, xlabel=xlabel, ylabel=ylabel, description=description
		)
		return TimeSeries2d(df, definition)

	@staticmethod
	def build_from_df(df, name, xname, yname, category=None, label=None, xlabel=None, ylabel=None, description=None):
		df.columns = [xname, yname]
		definition = DefinitionBuilder().build2d(
			name=name, category=category, object_type='TimeSeries2d',
			label=label, xlabel=xlabel, ylabel=ylabel, description=description
		)
		return TimeSeries2d(df.sort_index(), definition)
