from AnalyseClasses.AnalyseStarter import AnalyseStarter
from AnalyseClasses.Food.BaseFood import AnalyseBaseFood
from AnalyseClasses.Trajectory.BaseTrajectory import AnalyseTrajectory
from Scripts.root import root

group = 'UO'

starter = AnalyseStarter(root, group)
# starter.start(redo=False, index_dicts=True, markings=False, dynamic_food=True)

traj = AnalyseTrajectory(root, group)
# traj.initialize_xy_orientation_food(dynamic_food=True)
# traj.compute_speed()

BaseFood = AnalyseBaseFood(root, group)
# BaseFood.compute_speed_food()
# BaseFood.compute_distance_to_food()
# BaseFood.compute_is_xy_next_to_food()
# BaseFood.compute_xy_next_to_food()
# BaseFood.compute_speed_xy_next_to_food()
# BaseFood.compute_speed_next_to_food()
# BaseFood.compute_orientation_next_to_food()
# BaseFood.compute_orientation_to_food()
