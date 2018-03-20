import pandas as pd

from JsonFiles import JsonFiles


class Characteristics:
	def __init__(self, root, group):
		self.dict = JsonFiles.import_obj(root+group+'/Characteristics.json')

	def get_experiment(self, id_exp, name):
		return self.dict[int(id_exp)][name]

	def get_array(self, name):
		if name in ['session', 'trial', 'n_frames', 'fps', 'mm2px', 'food_radius']:
			return pd.DataFrame(
				[[key, self.dict[key][name]] for key in self.dict.keys()],
				columns=['id_exp', 'name'])
		elif name in ['food_center', 'traj_translation']:
			return pd.DataFrame(
				[[key]+self.dict[key][name] for key in self.dict.keys()],
				columns=['id_exp', 'x', 'y'])
		elif name in ['crop_limit_x', 'crop_limit_y']:
			return pd.DataFrame(
				[[key]+self.dict[key][name] for key in self.dict.keys()],
				columns=['id_exp', 'l_min', 'l_max'])
		elif name in ['entrance']:
			return pd.DataFrame(
				[[key]+self.dict[key][name][0]+self.dict[key][name][1]+self.dict[key][name][2] for key in self.dict.keys()],
				columns=['id_exp', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'])
		elif name in ['ref_pts']:
			return pd.DataFrame(
				[[key]+self.dict[key][name][0]+self.dict[key][name][1] for key in self.dict.keys()],
				columns=['id_exp', 'x1', 'y1', 'x2', 'y2'])
		else:
			print(name+' is not a characteristic')
