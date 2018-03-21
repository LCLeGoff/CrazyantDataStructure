import pandas as pd

from Characteristics import Characteristics
from JsonFiles import JsonFiles


class CharacteristicsLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'

	def build(self, name):
		chara_dict = JsonFiles.import_obj(self.root+name.category+'/Characteristics.json')
		if name.name in ['session', 'trial', 'n_frames', 'fps', 'mm2px', 'food_radius']:
			return Characteristics(pd.DataFrame(
				[[key, chara_dict[key][name.name]] for key in chara_dict.keys()],
				columns=['id_exp', 'name']), name)
		elif name.name in ['food_center', 'traj_translation']:
			return Characteristics(pd.DataFrame(
				[[key]+chara_dict[key][name.name] for key in chara_dict.keys()],
				columns=['id_exp', 'x', 'y']), name)
		elif name.name in ['crop_limit_x', 'crop_limit_y']:
			return Characteristics(pd.DataFrame(
				[[key]+chara_dict[key][name.name] for key in chara_dict.keys()],
				columns=['id_exp', 'l_min', 'l_max']), name)
		elif name.name in ['ref_pts']:
			return Characteristics(pd.DataFrame(
				[[key]+chara_dict[key][name.name][0]+chara_dict[key][name.name][1] for key in chara_dict.keys()],
				columns=['id_exp', 'x1', 'y1', 'x2', 'y2']), name)
		elif name.name in ['entrance']:
			return Characteristics(pd.DataFrame(
				[[key]+chara_dict[key][name.name][0]+chara_dict[key][name.name][1]+chara_dict[key][name.name][2]
					for key in chara_dict.keys()],
				columns=['id_exp', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3']), name)
		else:
			print(name+' is not a characteristic')