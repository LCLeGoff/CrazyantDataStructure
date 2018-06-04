
from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class RecruitmentDirection:
    def __init__(self, root, group):
        self.pd_idx_manager = PandasIndexManager()
        self.exp = ExperimentGroupBuilder(root).build(group)

    def compute_recruitment_direction(self):
        name_interval = 'recruitment_intervals'

        name_to_filter = 'xy_markings'
        result_name = 'xy_recruitments'
        self.exp.load([name_interval, name_to_filter])
        self.exp.filter_with_time_intervals(name_to_filter, name_interval, result_name)
        # self.exp.write(result_name)

        name_to_filter = 'phi_markings'
        result_name = 'phi_recruitments'
        self.exp.load([name_interval, name_to_filter])
        self.exp.filter_with_time_intervals(name_to_filter, name_interval, result_name)
        # self.exp.write(result_name)

        result_name = self.exp.compute_mean_in_time_interval(name_to_filter, name_interval)

        id_exp_list = self.exp.__dict__[result_name].get_index_array_of_id_exp()

        id_exp_ant_frame_array = self.exp.__dict__[result_name].get_index_array_of_id_exp_ant_frame()

        index_list = []
        for id_exp in id_exp_list:
            temp_index_array = id_exp_ant_frame_array[id_exp_ant_frame_array[:, 0] == id_exp, :]
            frame_array = temp_index_array[:, 2]
            idx_frame_min = frame_array.argmin()
            idx_min = tuple(temp_index_array[idx_frame_min, :])
            index_list.append(idx_min)

        self.exp.add_copy1d(
            name_to_copy=result_name, copy_name='first_recruitment'
        )
        self.exp.first_recruitment.df = self.exp.__dict__[result_name].get_row_of_idx_array(index_list)

    # self.exp.phi_markings_over_recruitment_intervals.plotter.hist1d(
    # 	title_prefix=self.exp.group, bins=np.arange(-1.125, 1.25, 0.25)*np.pi, marker='o', lw=1, normed=True)

# 	name_to_filter = 'phi_markings'
# 	result_name = 'phi_recruitments'
#
# exp.filter_with_time_intervals(
# 	'phi_markings', 'recruitment_intervals', 'phi_recruitments', label='', xlabel='', ylabel='', replace=True)
#
# 	self.co
