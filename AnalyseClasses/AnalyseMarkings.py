from Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from matplotlib import pyplot as plt


class AnalyseMarkings:
	def __init__(self, root, group):
		self.exp = ExperimentGroupBuilder(root).build(group)

	def spatial_repartition_markings(self):
		self.exp.load(['x', 'y', 'markings'])
		self.exp.filter(
			name1='x', name2='markings', new_name='x_markings',
			label='x', category='Markings', description='x coordinates of ant positions, while marking')
		self.exp.filter(
			name1='y', name2='markings', new_name='y_markings',
			label='y', category='Markings', description='y coordinates of ant positions, while marking')
		self.exp.to_2d(
			name1='x_markings', name2='y_markings', new_name='xy_markings',
			category='Markings', label='marking coordinates', xlabel='x', ylabel='y',
			description='coordinates of ant positions, while marking'
		)
		self.exp.to_2d(
			name1='x', name2='y', new_name='xy',
			category='Trajectory', label='Trajectories', xlabel='x', ylabel='y',
			description='coordinates of ant positions'
		)
		self.exp.xy.array.plot('x', 'y', style='o')
		self.exp.xy_markings.print()
		self.exp.xy_markings.array.plot('x_markings', 'y_markings', style='o')
		plt.show()
