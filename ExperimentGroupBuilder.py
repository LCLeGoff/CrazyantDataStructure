from ExperimentGroups import ExperimentGroups
from JsonFiles import JsonFiles
import numpy as np


class ExperimentGroupBuilder:
	def __init__(self, root):
		self.root = root

	def build(self, group):
		id_exp_list = np.array(list(JsonFiles.import_obj(self.root+group+'/Raw/Characteristics.json').keys()), dtype=int)
		return ExperimentGroups(self.root, group, id_exp_list)
