from ExperimentGroups import ExperimentGroups
from Tools.JsonFiles import import_id_exp_list


class ExperimentGroupBuilder:
	def __init__(self, root):
		self.root = root

	def build(self, group):
		id_exp_list = import_id_exp_list(self.root+group+'/')
		return ExperimentGroups(self.root, group, id_exp_list)
