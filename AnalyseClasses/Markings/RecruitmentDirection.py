from Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from PandasIndexManager.PandasIndexManager import PandasIndexManager


class RecruitmentDirection:
	def __init__(self, root, group):
		self.pd_idx_manager = PandasIndexManager()
		self.exp = ExperimentGroupBuilder(root).build(group)

	def compute_recruitment_direction(self):
		name_interval = 'recruitment_intervals'
		name_to_filter = 'xy_markings'
		result_name = 'xy_recruitments'
		self.exp.load([name_interval, name_to_filter])
		intervals_array = self.exp.__dict__[name_interval].convert_df_to_array()
		self.exp.add_new2d_empty(
			name=result_name, xname='x', yname='y',
			object_type='Events2d'
		)
		for id_exp, id_ant, t, dt in intervals_array:
			temp_xy_mark = self.exp.__dict__[name_to_filter].get_row_id_exp_ant_in_frame_interval(id_exp, id_ant, t, t+dt)
			self.exp.__dict__[result_name].add_rows(temp_xy_mark.index, temp_xy_mark[name_to_filter])
			# intervals.set_index(['id_exp', 'id_ant'], inplace=True)

		# interval
		# xy_mark =
