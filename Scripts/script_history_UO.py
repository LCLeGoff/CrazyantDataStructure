from AnalyseClasses.AnalyseStarter import AnalyseStarter
from AnalyseClasses.Food.FoodBase import AnalyseFoodBase
from AnalyseClasses.Food.FoodCarrying import AnalyseFoodCarrying
from AnalyseClasses.Trajectory.BaseTrajectory import AnalyseTrajectory
from Scripts.root import root

group = 'UO'

starter = AnalyseStarter(root, group)
# starter.start(redo=True, index_dicts=False, markings=False, dynamic_food=True)
# starter.compute_mm2px()

traj = AnalyseTrajectory(root, group)
# traj.initialize_xy_orientation_food(dynamic_food=True)
# traj.compute_speed()
# traj.compute_mm10_speed()
# traj.compute_mm20_speed()
# traj.compute_mm10_orientation()
# traj.compute_mm20_orientation()
# traj.compute_is_from_outside()

FoodBase = AnalyseFoodBase(root, group, exp=traj.exp)
# FoodBase.compute_food_phi()
# FoodBase.compute_speed_food()
#
# FoodBase.compute_distance2food()
# FoodBase.compute_mm10_distance2food()
# FoodBase.compute_mm20_distance2food()
#
# FoodBase.compute_is_xy_next2food()
# FoodBase.compute_xy_next2food()
#
# FoodBase.compute_speed_xy_next2food()
# FoodBase.compute_speed_next2food()
# FoodBase.compute_mm10_speed_next2food()
# FoodBase.compute_mm20_speed_next2food()
#
# FoodBase.compute_distance2food_next2food()
# FoodBase.compute_mm10_distance2food_next2food()
# FoodBase.compute_mm20_distance2food_next2food()

# FoodBase.compute_distance2food_next2food_differential()
# FoodBase.compute_mm10_distance2food_next2food_differential()
# FoodBase.compute_mm20_distance2food_next2food_differential()
#
# FoodBase.compute_orientation_next2food()
# FoodBase.compute_mm10_orientation_next2food()
# FoodBase.compute_mm20_orientation_next2food()
#
# FoodBase.compute_angle_body_food()
# FoodBase.compute_mm5_angle_body_food()
# FoodBase.compute_mm10_angle_body_food()
# FoodBase.compute_mm20_angle_body_food()
#
# FoodBase.compute_angle_body_food_next2food()
# FoodBase.compute_mm5_angle_body_food_next2food()
# FoodBase.compute_mm10_angle_body_food_next2food()
# FoodBase.compute_mm20_angle_body_food_next2food()


Carrying = AnalyseFoodCarrying(root, group, exp=FoodBase.exp)
# Carrying.compute_carrying_next2food_with_svm()
# Carrying.compute_carrying_from_svm()
# Carrying.compute_carrying()
# Carrying.compute_carried_food()
# Carrying.compute_carrying_intervals(True)
Carrying.compute_not_carrying_intervals(True)
