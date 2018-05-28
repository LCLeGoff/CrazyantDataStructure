from AnalyseClasses.Markings.BaseMarkings import AnalyseMarkings
from AnalyseClasses.AnalyseStarter import AnalyseStarter
from AnalyseClasses.Markings.Recruitment import Recruitment
from AnalyseClasses.Trajectory.BaseTrajectory import AnalyseTrajectory
from scripts.root import root

for group in ['FMAB', 'FMABU', 'FMABW']:
	print(group)
	# AnalyseStarter(root, group).start(False)
	# traj = AnalyseTrajectory(root, group)
	# traj.compute_x_y()
	# traj.compute_r_phi()
	#
	# mark = AnalyseMarkings(root, group)
	# mark.compute_xy_marking()
	# mark.compute_r_phi_marking()
	# mark.compute_marking_interval()
	# mark.compute_marking_distance()

	recruit = Recruitment(root, group)
	# recruit.compute_marking_batch()
	recruit.compute_recruitment()
