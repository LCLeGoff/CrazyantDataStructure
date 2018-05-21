import numpy as np

from pandas import IndexSlice as IdxSc
from Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from PandasIndexManager.PandasIndexManager import PandasIndexManager


class AnalyseMarkings:
	def __init__(self, root, group):
		self.pd_idx_manager = PandasIndexManager()
		self.exp = ExperimentGroupBuilder(root).build(group)

	def compute_xy_marking(self):
		print('xy_marking')
		self.exp.load(['x', 'y', 'markings'])

		self.exp.add_from_filtering(name_to_filter='x', name_filter='markings', result_name='x_markings')
		self.exp.add_from_filtering(name_to_filter='y', name_filter='markings', result_name='y_markings')

		self.exp.add_2d_from_1ds(
			name1='x_markings', name2='y_markings',
			result_name='xy_markings', xname='x', yname='y',
			category='Markings', label='marking coordinates', xlabel='x', ylabel='y',
			description='coordinates of ant positions, while marking'
		)

		self.exp.write('xy_markings')

	def compute_r_phi_marking(self):
		print('xy_marking_polar')
		self.exp.load(['r', 'phi', 'markings'])

		self.exp.add_from_filtering(
			name_to_filter='r', name_filter='markings', result_name='r_markings',
			category='Markings', label='marking radial coordinates',
			description='radial coordinates of ant positions, while marking')
		self.exp.write('r_markings')

		self.exp.add_from_filtering(
			name_to_filter='phi', name_filter='markings', result_name='phi_markings',
			category='Markings', label='marking angular coordinates',
			description='angular coordinates of ant positions, while marking')
		self.exp.write('phi_markings')

		self.exp.add_2d_from_1ds(
			name1='r_markings', name2='phi_markings',
			result_name='polar_markings', xname='r', yname='phi',
			category='Markings', label='marking polar coordinates', xlabel='r', ylabel='phi',
			description='polar coordinates of ant positions, while marking'
		)
		self.exp.write('polar_markings')

	def spatial_repartition_xy_markings(self):
		self.exp.load(['xy_markings'])
		self.exp.plot_repartition('xy_markings')

	def spatial_repartition_xy_markings_2d_hist(self):
		self.exp.load(['xy_markings'])
		self.exp.plot_repartition_hist('xy_markings')

	def compute_marking_interval(self):
		self.exp.load(['markings'])
		name = 'marking_interval'
		print(name)

		id_exp_ant_array = self.exp.markings.get_id_exp_ant_array()
		marking_interval_list = []
		for (id_exp, id_ant) in id_exp_ant_array:
			marks = self.get_marking_of_id_ant_id_exp(id_ant, id_exp)
			lg = len(marks)-1
			marking_interval_list = self.add_marking_intervals(marking_interval_list, id_ant, id_exp, lg, marks)

		marking_interval_df = self.pd_idx_manager.convert_to_exp_ant_frame_indexed_df(marking_interval_list, name)
		self.exp.add_new1d(
			array=marking_interval_df, name=name, object_type='Events1d', category='Markings',
			label='marking intervals', description='Time intervals between two marking events'
		)
		self.exp.write(name)

	@staticmethod
	def add_marking_intervals(marking_interval_list, id_ant, id_exp, lg, marks):
		id_exp_array = np.full(lg, id_exp)
		id_ant_array = np.full(lg, id_ant)
		mark_frames = marks[:-1, 2]
		marking_interval = marks[1:, 2] - marks[:-1, 2]
		marking_interval_list += list(zip(id_exp_array, id_ant_array, mark_frames, marking_interval))
		return marking_interval_list

	def get_marking_of_id_ant_id_exp(self, id_ant, id_exp):
		marks = np.array(self.exp.markings.get_row(IdxSc[id_exp, id_ant, :]).reset_index())
		return marks
