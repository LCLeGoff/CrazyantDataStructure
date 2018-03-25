import pandas as pd
import numpy as np

from Characteristics import Characteristics1d, Characteristics2d
from JsonFiles import JsonFiles


class CharacteristicsLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'

	def load(self, name):
		chara_dict = JsonFiles.import_obj(self.root+name.category+'/Characteristics.json')
		if name.name in ['session', 'trial', 'n_frames', 'fps', 'mm2px', 'food_radius']:
			res = pd.DataFrame(
				[chara_dict[key][name.name] for key in chara_dict.keys()],
				index=np.array(list(chara_dict.keys()), dtype=int),
				columns=[name.name])
			res.index.name = 'id_exp'
			return Characteristics1d(res, name)
		elif name.name in ['food_center', 'traj_translation', 'ref_pts1', 'ref_pts2', 'entrance1', 'entrance2']:
			res = pd.DataFrame(
				[np.array(chara_dict[key][name.name], dtype=float) for key in chara_dict.keys()],
				index=np.array(list(chara_dict.keys()), dtype=int),
				columns=['x', 'y'])
			res.index.name = 'id_exp'
			return Characteristics2d(res, name)
		elif name.name in ['crop_limit_x', 'crop_limit_y']:
			res = pd.DataFrame(
				[np.array(chara_dict[key][name.name], dtype=float) for key in chara_dict.keys()],
				index=np.array(list(chara_dict.keys()), dtype=int),
				columns=['l_min', 'l_max'])
			res.index.name = 'id_exp'
			return Characteristics2d(res, name)
		else:
			raise ValueError(name+' is not a characteristic')
