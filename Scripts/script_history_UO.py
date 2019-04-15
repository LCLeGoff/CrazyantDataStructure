from AnalyseClasses.AnalyseStarter import AnalyseStarter
from AnalyseClasses.Food.AntFoodRelation import AnalyseAntFoodRelation
from AnalyseClasses.Food.FoodBase import AnalyseFoodBase
from AnalyseClasses.Food.FoodCarrying import AnalyseFoodCarrying
from AnalyseClasses.Food.FoodVelocity import AnalyseFoodVelocity
from AnalyseClasses.Food.FoodEntropy import AnalyseFoodEntropy
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

AntFood = AnalyseAntFoodRelation(group, exp=Traj.exp)
# AntFood.compute_distance2food()

FoodBase = AnalyseFoodBase(group, exp=AntFood.exp)
# FoodBase.compute_food_traj_length()

# FoodBase.compute_food_speed(redo_hist=True)
# FoodBase.compute_food_speed_evol()

# FoodBase.compute_food_phi(True)

# FoodBase.compute_food_exit_angle()
# FoodBase.compute_mm1s_food_exit_angle()
# FoodBase.compute_mm10s_food_exit_angle()
# FoodBase.compute_mm30s_food_exit_angle()
# FoodBase.compute_mm60s_food_exit_angle()

# FoodBase.compute_food_exit_distance(True)
# FoodBase.compute_food_exit_distance_evol(True)

# FoodBase.compute_food_direction_error(True)
# FoodBase.compute_food_direction_error_evol(True)
# FoodBase.compute_mm1s_food_direction_error(True)
# FoodBase.compute_mm10s_food_direction_error(True)
# FoodBase.compute_mm30s_food_direction_error(True)


# ToDo: split this category in two
# FoodBase.compute_mm10_distance2food()
# FoodBase.compute_mm20_distance2food()
# FoodBase.exp.rename_data('mm10_distance2food', category='AntFoodRelation')
# FoodBase.exp.rename_data('mm20_distance2food', category='AntFoodRelation')
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

FoodVelocity = AnalyseFoodVelocity(group, exp=FoodBase.exp)
# FoodVelocity.compute_food_velocity(True)
# FoodVelocity.compute_food_velocity_phi_evol(True)
#
# FoodVelocity.compute_dotproduct_food_velocity_exit(True)
# FoodVelocity.compute_mm1s_dotproduct_food_velocity_exit(True)
# FoodVelocity.compute_mm10s_dotproduct_food_velocity_exit(redo_plot_indiv=True)
# FoodVelocity.compute_mm30s_dotproduct_food_velocity_exit(redo_plot_indiv=True)

# FoodVelocity.compute_mm1s_food_velocity_phi()

# FoodVelocity.compute_mm1s_food_velocity_vector()
# FoodVelocity.compute_mm10s_food_velocity_vector()
# FoodVelocity.compute_mm30s_food_velocity_vector()

# FoodVelocity.compute_mm1s_food_velocity_vector_length(redo=True)
# FoodVelocity.compute_mm10s_food_velocity_vector_length(redo=True)
# FoodVelocity.compute_mm30s_food_velocity_vector_length(redo_plot_indiv=True)


Carrying = AnalyseFoodCarrying(group, exp=FoodVelocity.exp)
# Carrying.compute_food_traj_length_around_first_attachment()
# Carrying.compute_outside_ant_attachment()
# Carrying.compute_non_outside_ant_attachment()
# Carrying.compute_nbr_attachment_per_exp()
# Carrying.compute_first_attachment_time_of_outside_ant()

# Carrying.compute_carrying_next2food_with_svm()
# Carrying.compute_carrying_from_svm()

# Carrying.compute_carrying()
# Carrying.compute_carried_food()
# Carrying.compute_carrying_intervals(True)
# Carrying.compute_not_carrying_intervals(True)

# Carrying.compute_food_direction_error_evol_around_first_attachment(True)
# Carrying.compute_autocorrelation_food_phi(True)
# Carrying.compute_autocorrelation_food_velocity_phi(True)
# Carrying.compute_autocorrelation_food_velocity_phi_indiv()

# Carrying.compute_food_phi_speed_entropy_evol_after_first_attachment(True)
# Carrying.compute_food_phi_speed_entropy_evol_per_exp(True)

# Carrying.compute_mm30s_dotproduct_food_velocity_exit_vs_food_velocity_vector_length()
# Carrying.compute_mm30s_food_direction_error_vs_food_velocity_vector_length()
# Carrying.compute_information_trajectory_around_attachment()

# Carrying.compute_w30s_entropy_mm1s_food_velocity_phi_evol_around_first_attachment()
# Carrying.compute_w30s_entropy_mm1s_food_velocity_phi_evol_around_outside_ant_attachments(True)

# Carrying.compute_mean_food_direction_error_around_outside_ant_attachments(True)
# Carrying.compute_mean_food_velocity_vector_length_around_outside_ant_attachments(True)

# Carrying.compute_mean_food_velocity_vector_length_vs_food_direction_error_around_outside_attachments()

FoodEntropy = AnalyseFoodEntropy(group, exp=Carrying.exp)
# FoodEntropy.compute_w30s_entropy_mm1s_food_velocity_phi_indiv_evol(True)
# FoodEntropy.compute_w1s_entropy_mm1s_food_velocity_phi_indiv_evol(True)
#
# FoodEntropy.compute_mm1s_food_direction_error_around_outside_attachments()
# FoodEntropy.compute_mm1s_food_direction_error_around_non_outside_attachments()
#
# FoodEntropy.compute_information_mm1s_food_direction_error_around_outside_attachments(True)
# FoodEntropy.compute_information_mm1s_food_direction_error_around_non_outside_attachments(True)
# FoodEntropy.compute_information_mm1s_food_direction_error_around_attachments(True)

# FoodEntropy.compute_information_mm1s_food_direction_error_around_first_outside_attachments()
FoodEntropy.compute_information_mm1s_food_direction_error_around_the_first_outside_attachment()
