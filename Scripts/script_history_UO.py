from AnalyseClasses.AnalyseStarter import AnalyseStarter
from AnalyseClasses.Trajectory.BaseTrajectory import AnalyseTrajectory
from Scripts.root import root

group = 'UO'

starter = AnalyseStarter(root, group)
starter.start(redo=True, markings=False)

traj = AnalyseTrajectory(root, group)
traj.compute_x_y()
# traj.compute_r_phi()

