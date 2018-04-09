from PandasIndexManager.PandasIndexManager import PandasIndexManager


class BuilderDataObject:
	def __init__(self, array):
		self.array = array
		self.array.sort_index(inplace=True)

	def operation(self, fct):
		self.array = fct(self.array)

	def print(self, short=True):
		if short:
			print(self.array.head())
		else:
			print(self.array)

	def get_value(self, idx):
		return self.array.loc[idx, :]

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

