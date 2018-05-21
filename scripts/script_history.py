from AnalyseClasses.Markings.BaseMarkings import AnalyseMarkings
from AnalyseClasses.AnalyseStarter import AnalyseStarter
from AnalyseClasses.Trajectory.BaseTrajectory import AnalyseTrajectory
from scripts.root import root

for group in ['FMAB', 'FMABU', 'FMABW']:
	print(group)
	# AnalyseStarter(root, group).start(False)
	traj = AnalyseTrajectory(root, group)
	traj.compute_x_y()
	traj.compute_r_phi()

	mark = AnalyseMarkings(root, group)
	mark.compute_xy_marking()
	mark.compute_r_phi_marking()
	mark.compute_marking_interval()
# 	# exp = ExperimentGroupBuilder(root).build(group)
# 	# exp.load('marking_interval')
# 	# exp.plot_hist1d('marking_interval', range(0, 1000), xscale='log', yscale='log', group=0)
# 	# mark.compute_first_marking_ant_radial_criterion(show=False)
# 	# mark.compute_first_marking_ant_batch_criterion(show=True)
# 	mark.compute_recruitment(show=False)
# 	# mark.compute_first_marking_ant_setup_orientation()
# 	# mark.compute_first_marking_ant_setup_orientation_circle()
# plt.show()
# #
#
# # for group in ['FMAB', 'FMABU', 'FMABW']:
# # 	mark = AnalyseMarkings(root, group)
# # 	# mark.spatial_repartition_xy_markings()
# # 	# mark.spatial_repartition_xy_markings_2d_hist()
# # 	# mark.spatial_repartition_first_markings()
# #
# # 	# mark.spatial_repartition_first_markings_2d_hist()
# # plt.show()
#
#
# for group in ['FMAB', 'FMABU', 'FMABW']:
# 	print(group)
# 	mark = AnalyseMarkings(root, group)
# 	# mark.compute_xy_marking()
# 	# mark.compute_xy_marking_polar()
# 	# mark.marking_interval()
