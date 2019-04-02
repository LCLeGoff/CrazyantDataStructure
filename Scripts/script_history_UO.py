from AnalyseClasses.AnalyseStarter import AnalyseStarter
from AnalyseClasses.Food.FoodBase import AnalyseFoodBase
from AnalyseClasses.Food.FoodCarrying import AnalyseFoodCarrying
from AnalyseClasses.Trajectory.BaseTrajectory import AnalyseTrajectory
from Scripts.root import root

group = 'UO'

starter = AnalyseStarter(root0=root, group=group)
# starter.start(redo=True, index_dicts=False, markings=False, dynamic_food=True)
# starter.compute_mm2px()
# starter.compute_exit0()


Traj = AnalyseTrajectory(group)
# Traj.initialize_xy_orientation_food(dynamic_food=True)
# Traj.compute_speed(redo_hist=True)
# Traj.compute_mm10_speed(redo_hist=True)
# Traj.compute_mm20_speed(redo_hist=True)
# Traj.compute_mm1s_speed(True)
# Traj.compute_mm10_orientation()
# Traj.compute_mm20_orientation()
# Traj.compute_is_from_outside()

FoodBase = AnalyseFoodBase(group, exp=Traj.exp)
# FoodBase.compute_food_traj_length()

# FoodBase.compute_food_phi(True)
# FoodBase.compute_food_exit_angle()
# FoodBase.compute_food_exit_distance(True)
# FoodBase.compute_food_exit_distance_evol(True)

# FoodBase.compute_food_velocity_phi(True)
# FoodBase.compute_food_velocity_phi_evol()

# FoodBase.compute_food_direction_error(True)
# FoodBase.compute_food_direction_error_evol(True)
# FoodBase.compute_mm1s_food_direction_error()

# FoodBase.compute_speed_food(redo_hist=True)

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


Carrying = AnalyseFoodCarrying(group, exp=FoodBase.exp)
# Carrying.compute_food_traj_length_around_first_attachment()
# Carrying.compute_outside_ant_attachment()
# Carrying.compute_first_attachment_time_of_outside_ant()

# Carrying.compute_carrying_next2food_with_svm()
# Carrying.compute_carrying_from_svm()

# Carrying.compute_carrying()
# Carrying.compute_carried_food()
# Carrying.compute_carrying_intervals(True)
# Carrying.compute_not_carrying_intervals(True)


# Carrying.compute_food_direction_error_evol_around_first_attachment(True)
# Carrying.compute_autocorrelation_food_phi(True)
# Carrying.compute_autocorrelation_food_velocity_phi()

# Carrying.compute_food_direction_error_entropy_evol_after_first_attachment(True)
Carrying.compute_food_direction_error_entropy_evol_per_exp(True)

# Carrying.compute_food_phi_speed_entropy_evol_after_first_attachment(True)
# Carrying.compute_food_phi_speed_entropy_evol_per_exp(True)
