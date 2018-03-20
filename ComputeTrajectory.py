from ExperimentGroupBuilder import ExperimentGroupBuilder


class ComputeTrajectory:
	def __init__(self, root, groups):
		experiment_builder = ExperimentGroupBuilder(root)
		self.groups = groups
		for group in groups:
			self.__dict__[group.lower()] = experiment_builder.build(group)

	def centered_x_y(self):
		self.fmab.load(['x', 'y'])