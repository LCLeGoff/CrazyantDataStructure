from AnalyseClasses.AnalyseStarter import AnalyseStarter
from AnalyseClasses.Food.FoodInformationTrajectory import AnalyseFoodInformationTrajectory
from AnalyseClasses.Food.AntFoodRelation import AnalyseAntFoodRelation
from AnalyseClasses.Food.FoodBase import AnalyseFoodBase
from AnalyseClasses.Food.FoodCarrying import AnalyseFoodCarrying
from AnalyseClasses.Food.FoodConfidence import AnalyseFoodConfidence
from AnalyseClasses.Food.FoodVelocity import AnalyseFoodVelocity
from AnalyseClasses.Food.FoodInformation import AnalyseFoodInformation
from AnalyseClasses.Food.FoodVeracity import AnalyseFoodVeracity
from AnalyseClasses.Trajectory.BaseTrajectory import AnalyseTrajectory
from Scripts.root import root

group = 'UO'

starter = AnalyseStarter(root0=root, group=group)
# starter.start(redo=True, index_dicts=False, markings=False, dynamic_food=True)
# starter.compute_mm2px()
# starter.compute_exit0()


Traj = AnalyseTrajectory(group)
# Traj.initialize_xy_orientation_food(dynamic_food=True)
# Traj.compute_mm10_traj()

# Traj.compute_speed(True)
# Traj.compute_mm10_speed(True)
# Traj.compute_mm20_speed(True)
# Traj.compute_mm1s_speed(True)
# Traj.compute_mm10_orientation()
# Traj.compute_mm20_orientation()
# Traj.compute_is_from_outside()

FoodBase = AnalyseFoodBase(group)
# FoodBase.compute_mm10_food_traj()
# FoodBase.compute_mm1s_food_traj()

# FoodBase.compute_food_traj_length()
# FoodBase.compute_food_first_frame()
# FoodBase.compute_food_exit_frames()
# FoodBase.compute_norm_time2frame(True)

# FoodBase.compute_food_speed(True)
# FoodBase.compute_food_speed_evol(True)

# FoodBase.compute_food_phi(True)

# FoodBase.compute_food_exit_angle()
# FoodBase.compute_mm1s_food_exit_angle()
# FoodBase.compute_mm10s_food_exit_angle()
# FoodBase.compute_mm30s_food_exit_angle()
# FoodBase.compute_mm60s_food_exit_angle()

# FoodBase.compute_food_exit_distance(True)
# FoodBase.compute_food_exit_distance_evol(True)


AntFood = AnalyseAntFoodRelation(group)
# AntFood.compute_distance2food()
# AntFood.compute_mm10_distance2food()
# AntFood.compute_mm20_distance2food()
#
# AntFood.compute_is_xy_next2food()
# AntFood.compute_xy_next2food()
#
# AntFood.compute_speed_xy_next2food()
# AntFood.compute_speed_next2food()
# AntFood.compute_mm10_speed_next2food()
# AntFood.compute_mm20_speed_next2food()
#
# AntFood.compute_distance2food_next2food()
# AntFood.compute_mm10_distance2food_next2food()
# AntFood.compute_mm20_distance2food_next2food()

# AntFood.compute_distance2food_next2food_diff()
# AntFood.compute_mm10_distance2food_next2food_diff()
# AntFood.compute_mm20_distance2food_next2food_diff()

# AntFood.compute_orientation_next2food()
# AntFood.compute_mm10_orientation_next2food()
# AntFood.compute_mm20_orientation_next2food()
#
# AntFood.compute_angle_body_food()
# AntFood.compute_mm10_angle_body_food()
# AntFood.compute_mm20_angle_body_food()

# AntFood.compute_angle_body_food_next2food()
# AntFood.compute_mm10_angle_body_food_next2food()
# AntFood.compute_mm20_angle_body_food_next2food()

Carrying = AnalyseFoodCarrying(group)
# Carrying.compute_carrying_next2food_with_svm()
# Carrying.compute_carrying_from_svm()

# Carrying.compute_carrying()
# Carrying.compute_carried_food()
# Carrying.compute_carrying_intervals(True)
# Carrying.compute_not_carrying_intervals(True)
# Carrying.compute_ant_attachments()
#
# Carrying.compute_outside_ant_carrying_intervals()
# Carrying.compute_outside_ant_attachments()
# Carrying.compute_non_outside_ant_carrying_intervals()

# Carrying.compute_isolated_ant_carrying_intervals()
# Carrying.compute_isolated_outside_ant_carrying_intervals()

# Carrying.compute_nbr_attachment_per_exp()
# Carrying.compute_first_attachment_time_of_outside_ant()
# Carrying.compute_food_traj_length_around_first_attachment()
# Carrying.compute_outside_ant_attachment_frames()
# Carrying.compute_non_outside_ant_attachment_frames()

# Carrying.compute_food_direction_error_evol_around_first_attachment(True)
# Carrying.compute_autocorrelation_food_phi(True)
# Carrying.compute_autocorrelation_food_velocity_phi(True)
# Carrying.compute_autocorrelation_food_velocity_phi_indiv(True)

# Carrying.compute_mm30s_dotproduct_food_velocity_exit_vs_food_velocity_vector_length()
# Carrying.compute_mm30s_food_direction_error_vs_food_velocity_vector_length()
# Carrying.compute_information_trajectory_around_attachment()

# Carrying.compute_mean_food_direction_error_around_outside_ant_attachments(True)
# Carrying.compute_mean_food_velocity_vector_length_around_outside_ant_attachments(True)

# Carrying.compute_mean_food_velocity_vector_length_vs_food_direction_error_around_outside_attachments()

FoodInfo = AnalyseFoodInformation(group)
# FoodInfo.compute_w30s_entropy_mm1s_food_velocity_phi_indiv_evol(True)
# FoodInfo.compute_w1s_entropy_mm1s_food_velocity_phi_indiv_evol(True)
#
#
# FoodInfo.compute_mm1s_food_direction_error_around_outside_attachments()
# FoodInfo.compute_mm1s_food_direction_error_around_non_outside_attachments()
#
#
# FoodInfo.compute_information_mm1s_food_direction_error_around_outside_attachments(redo_info=True)
# FoodInfo.compute_information_mm1s_food_direction_error_around_non_outside_attachments(True)
# FoodInfo.compute_information_mm1s_food_direction_error_around_attachments()
#
# FoodInfo.compute_information_mm1s_food_direction_error_around_first_outside_attachments()
# FoodInfo.compute_information_mm1s_food_direction_error_around_the_first_outside_attachment(True)
#
#
# FoodInfo.compute_fisher_information_mm1s_food_direction_error_around_outside_attachments(redo_info=True)
# FoodInfo.compute_fisher_information_mm1s_food_direction_error_around_non_outside_attachments(redo_info=True)
# FoodInfo.compute_fisher_information_mm1s_food_direction_error_around_attachments(redo_info=True)
#
# FoodInfo.compute_fisher_information_mm1s_food_direction_error_around_first_outside_attachments()
# FoodInfo.compute_fisher_information_mm1s_food_direction_error_around_the_first_outside_attachment()

FoodVelocity = AnalyseFoodVelocity(group)
# FoodVelocity.compute_food_velocity(True)
# FoodVelocity.compute_food_velocity_phi_evol(True)
#
# FoodVelocity.compute_mm1s_food_velocity_phi()
#
# FoodVelocity.compute_mm1s_food_velocity_vector()
# FoodVelocity.compute_mm10s_food_velocity_vector()
# FoodVelocity.compute_mm30s_food_velocity_vector()
#
# FoodVelocity.compute_mm1s_food_velocity_vector_length(True)
# FoodVelocity.compute_mm10s_food_velocity_vector_length(True)
# FoodVelocity.compute_mm30s_food_velocity_vector_length(True)
#
# FoodVelocity.compute_dotproduct_food_velocity_exit(True)
# FoodVelocity.compute_mm1s_dotproduct_food_velocity_exit(True)
# FoodVelocity.compute_mm10s_dotproduct_food_velocity_exit(True)
# FoodVelocity.compute_mm30s_dotproduct_food_velocity_exit(True)

FoodConfidence = AnalyseFoodConfidence(group)
# FoodConfidence.compute_w10s_food_crossed_distance(True)
# FoodConfidence.compute_w30s_food_crossed_distance(True)
#
# FoodConfidence.compute_w10s_food_total_crossed_distance(True)
# FoodConfidence.compute_w30s_food_total_crossed_distance(True)
#
# FoodConfidence.compute_w10s_food_path_efficiency(True)
# FoodConfidence.compute_w30s_food_path_efficiency(True)

FoodVeracity = AnalyseFoodVeracity(group)
# FoodVeracity.compute_food_direction_error(True)
# FoodVeracity.compute_food_direction_error_evol(True)
# FoodVeracity.compute_mm1s_food_direction_error(True)
# FoodVeracity.compute_mm10s_food_direction_error(True)
# FoodVeracity.compute_mm30s_food_direction_error(True)

FoodInfoTraj = AnalyseFoodInformationTrajectory(group)
# FoodInfoTraj.compute_w10s_food_direction_error_vs_path_efficiency()
# FoodInfoTraj.compute_w30s_food_direction_error_vs_path_efficiency()

# FoodInfoTraj.w10s_food_direction_error_vs_path_efficiency_hist2d(True)
# FoodInfoTraj.w30s_food_direction_error_vs_path_efficiency_hist2d(True)
#
# FoodInfoTraj.w10s_food_direction_error_vs_path_efficiency_hist2d_around_first_outside_attachment(True)
# FoodInfoTraj.w30s_food_direction_error_vs_path_efficiency_hist2d_around_first_outside_attachment(True)
#
# FoodInfoTraj.w10s_food_direction_error_vs_path_efficiency_hist2d_around_first_outside_attachment_norm_time(True)
# FoodInfoTraj.w30s_food_direction_error_vs_path_efficiency_hist2d_around_first_outside_attachment_norm_time(True)
#
# FoodInfoTraj.w10s_smooth_food_direction_error_vs_path_efficiency_scatter_around_first_outside_attachment(True)

# FoodInfoTraj.w10s_food_direction_error_vs_path_efficiency_velocity()
# FoodInfoTraj.w30s_food_direction_error_vs_path_efficiency_velocity()

# FoodInfoTraj.w10s_food_direction_error_vs_path_efficiency_vector_field()
# FoodInfoTraj.w30s_food_direction_error_vs_path_efficiency_vector_field()

# FoodInfoTraj.w10s_food_direction_error_vs_path_efficiency_probability_matrix(True)
# FoodInfoTraj.w30s_food_direction_error_vs_path_efficiency_probability_matrix(True)
