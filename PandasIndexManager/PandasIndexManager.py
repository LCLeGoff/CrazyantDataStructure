import numpy as np


class PandasIndexManager:
	def __init__(self):
		pass

	@staticmethod
	def get_id_exp_array(array):
		arr = array.reset_index()
		if 'id_exp' not in arr.columns:
			raise IndexError('array does not have id_exp as index')
		else:
			return np.array([idx for idx in set([idx for idx in arr.set_index(['id_exp']).index])])

	@staticmethod
	def get_id_exp_ant_array(array):
		arr = array.reset_index()
		if 'id_exp' not in arr.columns or 'id_ant' not in arr.columns:
			raise IndexError('array does not have id_exp or id_ant as index')
		else:
			idx_set = set([idx for idx in arr.set_index(['id_exp', 'id_ant']).index])
			return np.array([[id_exp, id_ant] for (id_exp, id_ant) in idx_set])

	def get_id_exp_ant_dict(self, array):
		index_array = self.get_id_exp_ant_array(array)
		res = dict()
		for (id_exp, id_ant) in index_array:
			if id_exp in res:
				res[id_exp].append(id_ant)
			else:
				res[id_exp] = [id_ant]
		return res

	@staticmethod
	def get_id_exp_ant_frame_array(array):
		arr = array.reset_index()
		if 'id_exp' not in arr.columns or 'id_ant' not in arr.columns or 'frame' not in arr.columns:
			raise IndexError('array does not have id_exp or id_ant or frame as index')
		else:
			idx_set = set([idx for idx in arr.set_index(['id_exp', 'id_ant', 'frame']).index])
			return np.array([[id_exp, id_ant, frame] for (id_exp, id_ant, frame) in idx_set])

	def get_id_exp_ant_frame_dict(self, array):
		index_array = self.get_id_exp_ant_array(array)
		res = dict()
		for (id_exp, id_ant, frame) in index_array:
			if id_exp in res:
				if id_ant in res[id_exp]:
					res[id_exp][id_exp].append(frame)
				else:
					res[id_exp][id_exp] = [frame]
			else:
				res[id_exp] = dict()
				res[id_exp][id_ant] = [frame]
		return res