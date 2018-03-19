from ExperimentGroups import ExperimentGroups


class ExperimentGroupBuilder:
	def __init__(self, root):
		self.root = root

	def build(self, group):
		return ExperimentGroups(self.root, group)
