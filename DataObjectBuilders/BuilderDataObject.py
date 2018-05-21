import pandas as pd
import numpy as np

from PandasIndexManager.PandasIndexManager import PandasIndexManager


class BuilderDataObject:
	def __init__(self, array):
		self.array = array
		# self.array.sort_index(inplace=True)

	def operation(self, fct):
		self.array = fct(self.array)

	def print(self, short=True):
		if short:
			print(self.array.head())
		else:
			print(self.array)

	def convert_df_to_array(self):
		return np.array(self.array.reset_index())

	def get_row(self, idx):
		return self.array.loc[idx, :]

	def get_row_id_exp(self, id_exp):
		return self.array.loc[pd.IndexSlice[id_exp, :, :], :]

	def get_row_id_exp_ant(self, id_exp, id_ant):
		return self.array.loc[pd.IndexSlice[id_exp, id_ant, :], :]

	def get_row_id_exp_ant_frame_from_array(self, idx_arr):
		return self.array.loc[list(map(tuple, np.array(idx_arr))), :]

	def get_row_id_exp_ant_frame(self, id_exp, id_ant, frame):
		return self.array.loc[pd.IndexSlice[id_exp, id_ant, frame], :]

	def get_row_id_exp_ant_in_frame_interval(self, id_exp, id_ant, frame0=None, frame1=None):
		if frame0 is None:
			frame0 = 0
		if frame1 is None:
			return self.array.loc[pd.IndexSlice[id_exp, id_ant, frame0:], :]
		else:
			return self.array.loc[pd.IndexSlice[id_exp, id_ant, frame0:frame1], :]

	def get_id_exp_array(self):
		return PandasIndexManager().get_id_exp_array(self.array)

	def get_id_exp_ant_array(self):
		return PandasIndexManager().get_id_exp_ant_array(self.array)

	def get_id_exp_ant_frame_array(self):
		return PandasIndexManager().get_id_exp_ant_frame_array(self.array)

	def get_id_exp_ant_dict(self):
		return PandasIndexManager().get_id_exp_ant_dict(self.array)

	def get_id_exp_ant_frame_dict(self):
		return PandasIndexManager().get_id_exp_ant_frame_dict(self.array)

	def add_row(self, idx, value, replace=False):
		if replace is False and idx in self.array.index:
			raise IndexError('Index '+str(idx)+' already exists')
		else:
			self.array.loc[idx] = value
