import pandas as pd
import numpy as np


class PandasIndexManager:
	def __init__(self):
		pass

	@staticmethod
	def convert_to_exp_ant_frame_indexed_df(array, name):
		df = pd.DataFrame(array, columns=['id_exp', 'id_ant', 'frame', name])
		df.set_index(['id_exp', 'id_ant', 'frame'], inplace=True)
		return df

	@staticmethod
	def get_array_id_exp(df):
		arr = df.reset_index()
		if 'id_exp' not in arr.columns:
			raise IndexError('array does not have id_exp as index')
		else:
			idx_set = set([idx for idx in arr.set_index(['id_exp']).index])
			return np.array(sorted(idx_set), dtype=int)

	@staticmethod
	def get_array_id_exp_ant(df):
		arr = df.reset_index()
		if 'id_exp' not in arr.columns or 'id_ant' not in arr.columns:
			raise IndexError('array does not have id_exp or id_ant as index')
		else:
			idx_set = set([idx for idx in arr.set_index(['id_exp', 'id_ant']).index])
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
	def get_array_id_exp_ant_frame(df):
		arr = df.reset_index()
		if 'id_exp' not in arr.columns or 'id_ant' not in arr.columns or 'frame' not in arr.columns:
			raise IndexError('array does not have id_exp or id_ant or frame as index')
		else:
			idx_set = set([idx for idx in arr.set_index(['id_exp', 'id_ant', 'frame']).index])
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
