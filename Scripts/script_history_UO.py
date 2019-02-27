from AnalyseClasses.AnalyseStarter import AnalyseStarter
from AnalyseClasses.Food.FoodBase import AnalyseFoodBase
from AnalyseClasses.Trajectory.BaseTrajectory import AnalyseTrajectory
from Scripts.root import root

group = 'UO'

starter = AnalyseStarter(root, group)
# starter.start(redo=False, index_dicts=True, markings=False, dynamic_food=True)

traj = AnalyseTrajectory(root, group)
# traj.initialize_xy_orientation_food(dynamic_food=True)
# traj.compute_speed()
# traj.compute_mm1s_speed()

FoodBase = AnalyseFoodBase(root, group)
# FoodBase.compute_speed_food()
# FoodBase.compute_distance_to_food()
# FoodBase.compute_is_xy_next_to_food()
# FoodBase.compute_xy_next_to_food()
# FoodBase.compute_speed_xy_next_to_food()

# FoodBase.compute_speed_next_to_food()
# FoodBase.compute_mm5min_speed_next2food()
# FoodBase.compute_mm10s_speed_next2food()
# FoodBase.compute_mm2s_speed_next2food()
# FoodBase.compute_mm1s_speed_next2food()
# FoodBase.compute_mm10_speed_next2food()

# FoodBase.compute_distance_to_food_next_to_food()
# FoodBase.compute_mm5min_distance2food_next2food()
# FoodBase.compute_mm60s_distance2food_next2food()
# FoodBase.compute_mm10s_distance2food_next2food()
# FoodBase.compute_mm1s_distance2food_next2food()
# FoodBase.compute_mm10_distance2food_next2food()

# FoodBase.compute_distance_to_food_next_to_food_differential()
# FoodBase.compute_mm5min_distance2food_next2food_diff()
# FoodBase.compute_mm60s_distance2food_next2food_diff()
# FoodBase.compute_mm10s_distance2food_next2food_diff()
# FoodBase.compute_mm2s_distance2food_next2food_diff()
# FoodBase.compute_mm1s_distance2food_next2food_diff()
# FoodBase.compute_mm10_distance2food_next2food_diff()
# FoodBase.compute_mm3_distance2food_next2food_diff()

# FoodBase.compute_orientation_next_to_food()
# FoodBase.compute_orientation_to_food()
FoodBase.compute_mm5min_orientation2food()
# FoodBase.compute_mm60s_orientation2food()
# FoodBase.compute_mm10s_orientation2food()
# FoodBase.compute_mm1s_orientation2food()
# FoodBase.compute_mm10_orientation2food()

FoodBase.compute_is_carrying()

