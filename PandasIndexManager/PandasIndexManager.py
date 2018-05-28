import pandas as pd
import numpy as np


class PandasIndexManager:
	def __init__(self):
		pass

	@staticmethod
	def create_empty_exp_indexed_df(name):
		df = pd.DataFrame(columns=['id_exp', name]).set_index(['id_exp'])
		return df.sort_index()

	@staticmethod
	def create_empty_exp_ant_indexed_df(name):
		df = pd.DataFrame(columns=['id_exp', 'id_ant', name]).set_index(['id_exp', 'id_ant'])
		return df.sort_index()

	@staticmethod
	def create_empty_exp_ant_frame_indexed_1d_df(name):
		df = pd.DataFrame(columns=['id_exp', 'id_ant', 'frame', name]).set_index(['id_exp', 'id_ant', 'frame'])
		return df.sort_index()

	@staticmethod
	def create_empty_exp_ant_frame_indexed_2d_df(xname, yname):
		df = pd.DataFrame(columns=['id_exp', 'id_ant', 'frame', xname, yname]).set_index(['id_exp', 'id_ant', 'frame'])
		return df.sort_index()

	@staticmethod
	def convert_to_exp_indexed_df(array, name):
		df = pd.DataFrame(array, columns=['id_exp', name])
		df.set_index(['id_exp'], inplace=True)
		return df.sort_index()

	@staticmethod
	def convert_to_exp_ant_indexed_df(array, name):
		df = pd.DataFrame(array, columns=['id_exp', 'id_ant', name])
		df.set_index(['id_exp', 'id_ant'], inplace=True)
		return df.sort_index()

	@staticmethod
	def convert_to_exp_ant_frame_indexed_1d_df(array, name):
		df = pd.DataFrame(array, columns=['id_exp', 'id_ant', 'frame', name])
		df.set_index(['id_exp', 'id_ant', 'frame'], inplace=True)
		return df.sort_index()

	@staticmethod
	def convert_to_exp_ant_frame_indexed_2d_df(array, xname, yname):
		df = pd.DataFrame(array, columns=['id_exp', 'id_ant', 'frame', xname, yname])
		df.set_index(['id_exp', 'id_ant', 'frame'], inplace=True)
		return df.sort_index()

	@staticmethod
	def get_array_id_exp(df_arg):
		df = df_arg.reset_index()
		if 'id_exp' not in df.columns:
			raise IndexError('df does not have id_exp as index')
		else:
			idx_set = set([idx for idx in df.set_index(['id_exp']).index])
			return np.array(sorted(idx_set), dtype=int)

	@staticmethod
	def get_array_id_exp_ant(df_arg):
		df = df_arg.reset_index()
		if 'id_exp' not in df.columns or 'id_ant' not in df.columns:
			raise IndexError('df does not have id_exp or id_ant as index')
		else:
			idx_set = set([idx for idx in df.set_index(['id_exp', 'id_ant']).index])
			return np.array(sorted(idx_set), dtype=int)

	def get_dict_id_exp_ant(self, df):
		index_array = self.get_array_id_exp_ant(df)
		res = dict()
		for (id_exp, id_ant) in index_array:
			if id_exp in res:
				res[id_exp].append(id_ant)
			else:
				res[id_exp] = [id_ant]
		for id_exp in res:
			res[id_exp].sort()
		return res

	@staticmethod
	def get_array_id_exp_ant_frame(df_arg):
		df = df_arg.reset_index()
		if 'id_exp' not in df.columns or 'id_ant' not in df.columns or 'frame' not in df.columns:
			raise IndexError('df does not have id_exp or id_ant or frame as index')
		else:
			idx_set = set([idx for idx in df.set_index(['id_exp', 'id_ant', 'frame']).index])
			return np.array(sorted(idx_set), dtype=int)

	def get_dict_id_exp_ant_frame(self, df):
		index_array = self.get_array_id_exp_ant_frame(df)
		res = dict()
		for (id_exp, id_ant, frame) in index_array:
			if id_exp in res:
				if id_ant in res[id_exp]:
					res[id_exp][id_ant].append(frame)
				else:
					res[id_exp][id_ant] = [frame]
			else:
				res[id_exp] = dict()
				res[id_exp][id_ant] = [frame]
		for id_exp in res:
			for id_ant in res[id_exp]:
				res[id_exp][id_ant].sort()
		return res
