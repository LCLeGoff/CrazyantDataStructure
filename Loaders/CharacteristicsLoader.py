import pandas as pd
import numpy as np

from Models.Characteristics import Characteristics1d, Characteristics2d
from Tools.JsonFiles import JsonFiles


class Characteristics1dLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'

	def load(self, name):
		chara_dict = JsonFiles.import_obj(self.root+name.category+'/Characteristics.json')
		res = pd.DataFrame(
			[chara_dict[key][name.name] for key in chara_dict.keys()],
			index=np.array(list(chara_dict.keys()), dtype=int),
			columns=[name.name])
		res.index.name = 'id_exp'
		return Characteristics1d(res, name)


class Characteristics2dLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'

	def load(self, name):
		chara_dict = JsonFiles.import_obj(self.root+name.category+'/Characteristics.json')
		res = pd.DataFrame(
			[np.array(chara_dict[key][name.name], dtype=float) for key in chara_dict.keys()],
			index=np.array(list(chara_dict.keys()), dtype=int),
			columns=['x', 'y'])
		res.index.name = 'id_exp'
		return Characteristics2d(res, name)
