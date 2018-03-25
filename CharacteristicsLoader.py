import pandas as pd
import numpy as np

from Characteristics import Characteristics
from JsonFiles import JsonFiles


class CharacteristicsLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'

	def build(self, name):
		chara_dict = JsonFiles.import_obj(self.root+name.category+'/Characteristics.json')
		if name.name in ['session', 'trial', 'n_frames', 'fps', 'mm2px', 'food_radius']:
			res = pd.DataFrame(
				[chara_dict[key][name.name] for key in chara_dict.keys()],
				index=np.array(list(chara_dict.keys()), dtype=int),
				columns=[name.name])
			res.index.name = 'id_exp'
		elif name.name in ['food_center', 'traj_translation']:
			res = pd.DataFrame(
				[np.array(chara_dict[key][name.name], dtype=float) for key in chara_dict.keys()],
				index=np.array(list(chara_dict.keys()), dtype=int),
				columns=['x', 'y'])
			res.index.name = 'id_exp'
		elif name.name in ['crop_limit_x', 'crop_limit_y']:
			res = pd.DataFrame(
				[np.array(chara_dict[key][name.name], dtype=float) for key in chara_dict.keys()],
				index=np.array(list(chara_dict.keys()), dtype=int),
				columns=['l_min', 'l_max'])
			res.index.name = 'id_exp'
		elif name.name in ['ref_pts']:
			res = pd.DataFrame(
				[np.array(chara_dict[key][name.name][0]+chara_dict[key][name.name][1], dtype=float) for key in chara_dict.keys()],
				index=np.array(list(chara_dict.keys()), dtype=int),
				columns=['x1', 'y1', 'x2', 'y2'])
			res.index.name = 'id_exp'
		elif name.name in ['entrance']:
			res = pd.DataFrame(
				[np.array(chara_dict[key][name.name][0]+chara_dict[key][name.name][1]+chara_dict[key][name.name][2], dtype=float)
					for key in chara_dict.keys()],
				index=np.array(list(chara_dict.keys()), dtype=int),
				columns=['x1', 'y1', 'x2', 'y2', 'x3', 'y3'])
			res.index.name = 'id_exp'
		else:
			raise ValueError(name+' is not a characteristic')

		return Characteristics(res, name)
