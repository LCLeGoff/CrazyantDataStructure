from AnalyseClasses.AnalyseStarter import AnalyseStarter
from AnalyseClasses.Food.BaseFood import AnalyseBaseFood
from AnalyseClasses.Trajectory.BaseTrajectory import AnalyseTrajectory
from Scripts.root import root

group = 'UO'

starter = AnalyseStarter(root, group)
# starter.start(redo=False, markings=False, dynamic_food=True)

traj = AnalyseTrajectory(root, group)
# traj.initialize_xy_orientation_food(dynamic_food=True)

BaseFood = AnalyseBaseFood(root, group)
BaseFood.compute_traj_next_food()

