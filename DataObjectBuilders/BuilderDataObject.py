import pandas as pd
import numpy as np

from PandasIndexManager.PandasIndexManager import PandasIndexManager


class BuilderDataObject:
	def __init__(self, df):
		self.df = df
		# self.df.sort_index(inplace=True)

	def operation(self, fct):
		self.df = fct(self.df)

	def print(self, short=True):
		if short:
			print(self.df.head())
		else:
			print(self.df)

	def convert_df_to_array(self):
		return np.array(self.df.reset_index())

	def get_row(self, idx):
		return self.df.loc[idx, :]

	def get_row_of_idx_array(self, idx_array):
		return self.df.loc[list(map(tuple, np.array(idx_array))), :]

	def get_index_array_of_id_exp(self):
		return PandasIndexManager().get_array_id_exp(self.df)

	def get_index_array_of_id_exp_ant(self):
		return PandasIndexManager().get_array_id_exp_ant(self.df)

	def get_index_array_of_id_exp_ant_frame(self):
		return PandasIndexManager().get_array_id_exp_ant_frame(self.df)

	def get_index_dict_of_id_exp_ant(self):
		return PandasIndexManager().get_dict_id_exp_ant(self.df)

	def add_row(self, idx, value, replace=False):
		if replace is False and idx in self.df.index:
			raise IndexError('Index '+str(idx)+' already exists')
		else:
			self.df.loc[idx] = value

	def add_rows(self, idx_list, value_list, replace=False):
		if len(idx_list) == len(value_list):
			for ii in range(len(idx_list)):
				self.add_row(idx_list[ii], value_list[ii], replace=replace)
		else:
			raise IndexError('Index and value list not same lengths')
