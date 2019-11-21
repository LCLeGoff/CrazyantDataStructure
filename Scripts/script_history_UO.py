from AnalyseClasses.AnalyseStarter import AnalyseStarter
from AnalyseClasses.CleaningData import CleaningData
from AnalyseClasses.Food.FoodInformationTrajectory import AnalyseFoodInformationTrajectory
from AnalyseClasses.Food.AntFoodRelation import AnalyseAntFoodRelation
from AnalyseClasses.Food.FoodBase import AnalyseFoodBase
from AnalyseClasses.Food.FoodCarrying import AnalyseFoodCarrying
from AnalyseClasses.Food.FoodConfidence import AnalyseFoodConfidence
from AnalyseClasses.Food.FoodVelocity import AnalyseFoodVelocity
from AnalyseClasses.Food.FoodInformation import AnalyseFoodInformation
from AnalyseClasses.Food.FoodVeracity import AnalyseFoodVeracity
from AnalyseClasses.Food.LeaderFollower import AnalyseLeaderFollower
from AnalyseClasses.Food.SVMFeatures import AnalyseSVMFeatures
from AnalyseClasses.Trajectory.BaseTrajectory import AnalyseTrajectory
from Scripts.root import root

group = 'UO'

starter = AnalyseStarter(root0=root, group=group)
# starter.start(redo=True, markings=False, dynamic_food=True)
# starter.compute_mm2px()
# starter.compute_exit0()
# starter.compute_is_from_outside()

CleaningData = CleaningData(root=root, group=group)
# CleaningData.interpolate_xy_orientation_food(dynamic_food=True)
# CleaningData.initialize_xy_orientation_food(dynamic_food=True)
# starter.compute_is_from_outside()

Traj = AnalyseTrajectory(group)
# Traj.compute_mm10_bodyLength()
# Traj.compute_mm10_traj()
# Traj.compute_speed(True)
# Traj.compute_mm10_speed()
# Traj.compute_mm20_speed(True)
# Traj.compute_mm1s_speed(True)
# Traj.compute_mm10_orientation()
# Traj.compute_mm20_orientation()

FoodBase = AnalyseFoodBase(group)
# FoodBase.compute_mm10_food_traj()
# FoodBase.compute_mm1s_food_traj()

# FoodBase.compute_food_traj_length()
# FoodBase.compute_food_first_frame()
# FoodBase.compute_food_exit_frames()
# FoodBase.compute_norm_time2frame(True)

# FoodBase.compute_food_speed(redo_hist=True)
# FoodBase.compute_mm10_food_speed()
# FoodBase.compute_mm1s_food_speed()
# FoodBase.compute_food_speed_evol(True)

# FoodBase.compute_food_phi(True)
#
# FoodBase.compute_food_exit_angle()
# FoodBase.compute_mm10_food_exit_angle()
# FoodBase.compute_mm1s_food_exit_angle()
# FoodBase.compute_mm10s_food_exit_angle()
# FoodBase.compute_mm30s_food_exit_angle()
# FoodBase.compute_mm60s_food_exit_angle()
#
# FoodBase.compute_food_exit_distance(True)
# FoodBase.compute_food_exit_distance_evol(True)
# FoodBase.compute_food_border_distance(True)

# FoodBase.compute_food_r(True)
# FoodBase.compute_food_r_mean_evol(True)
# FoodBase.compute_food_r_mean_evol_around_first_attachment(True)
# FoodBase.compute_food_r_var_evol(True)
# FoodBase.compute_food_r_var_evol_around_first_attachment(True)
# FoodBase.compute_food_phi_var_evol(True)
# FoodBase.compute_food_phi_var_evol_around_first_attachment(True)

# FoodBase.compute_food_phi_hist_evol(True)
# FoodBase.compute_food_phi_hist_evol_around_first_outside_attachment(True)


SVMFeatures = AnalyseSVMFeatures(group)
# SVMFeatures.compute_ant_body_end()
# SVMFeatures.compute_distance2food()
# SVMFeatures.compute_mm10_distance2food()
# SVMFeatures.compute_mm20_distance2food()
#
# SVMFeatures.compute_is_xy_next2food()
# SVMFeatures.compute_xy_next2food()
#
# SVMFeatures.compute_speed_xy_next2food()
# SVMFeatures.compute_speed_next2food()
# SVMFeatures.compute_mm10_speed_next2food()
# SVMFeatures.compute_mm20_speed_next2food()
#
# SVMFeatures.compute_distance2food_next2food()
# SVMFeatures.compute_mm10_distance2food_next2food()
# SVMFeatures.compute_mm20_distance2food_next2food()
#
# SVMFeatures.compute_distance2food_next2food_diff()
# SVMFeatures.compute_mm10_distance2food_next2food_diff()
# SVMFeatures.compute_mm20_distance2food_next2food_diff()
#
# SVMFeatures.compute_orientation_next2food()
# SVMFeatures.compute_mm10_orientation_next2food()
# SVMFeatures.compute_mm20_orientation_next2food()
#
# SVMFeatures.compute_angle_body_food()
# SVMFeatures.compute_mm10_angle_body_food()
# SVMFeatures.compute_mm20_angle_body_food()
#
# SVMFeatures.compute_angle_body_food_next2food()
# SVMFeatures.compute_mm10_angle_body_food_next2food()
# SVMFeatures.compute_mm20_angle_body_food_next2food()
#
# SVMFeatures.compute_food_angular_component_ant_velocity()
# SVMFeatures.compute_attachment_xy()
# SVMFeatures.compute_mm10_attachment_xy()
#
# SVMFeatures.compute_ant_food_phi()

Carrying = AnalyseFoodCarrying(group)
# Carrying.compute_carrying_next2food_with_svm()
# Carrying.compute_carrying_from_svm()
# Carrying.compute_food_rotation(True)
# Carrying.compute_mm10_food_rotation()
# Carrying.compute_ant_angular_speed()
# Carrying.compute_ant_food_relative_angular_speed()
# Carrying.compute_carrying_next2food_with_svm_with_angular_speed()
# Carrying.compute_carrying()

# Carrying.compute_food_rotation_evol(True)
# Carrying.compute_food_rotation_evol_around_first_outside_attachment(True)
# Carrying.compute_food_rotation_variance_evol(True)
# Carrying.compute_food_rotation_variance_evol_around_first_outside_attachment(True)

# Carrying.compute_food_rotation_acceleration(True)

# Carrying.compute_carried_food()
# Carrying.compute_carrying_intervals(True)
# Carrying.compute_not_carrying_intervals(True)
# Carrying.compute_aviram_carrying_intervals(True)

# Carrying.compute_outside_ant_carrying_intervals(True)
# Carrying.compute_non_outside_ant_carrying_intervals(True)
#
# Carrying.compute_ant_attachments()
# Carrying.compute_outside_ant_attachments()
# Carrying.compute_non_outside_ant_attachments()
# Carrying.compute_attachment_intervals(True)
# Carrying.compute_outside_attachment_intervals(True)
# Carrying.compute_non_outside_attachment_intervals(True)
#
# Carrying.compute_isolated_ant_carrying_intervals()
# Carrying.compute_isolated_outside_ant_carrying_intervals()
# Carrying.compute_isolated_non_outside_ant_carrying_intervals()

# Carrying.compute_nbr_attachment_per_exp()
# Carrying.compute_first_attachment_time_of_outside_ant()
# Carrying.compute_food_traj_length_around_first_attachment()
# Carrying.compute_outside_ant_attachment_frames()
# Carrying.compute_non_outside_ant_attachment_frames()
#
# Carrying.compute_mean_food_direction_error_around_outside_ant_attachments(True)
# Carrying.compute_mean_food_velocity_vector_length_around_outside_ant_attachments(True)
# Carrying.compute_mean_food_velocity_vector_length_vs_food_direction_error_around_outside_attachments(True)

# Carrying.compute_nb_carriers(True)
# Carrying.compute_nb_outside_carriers(True)
# Carrying.compute_nb_carriers_mean_evol_around_first_attachment(True)
# Carrying.compute_food_rotation_vs_nb_carriers(True)
# Carrying.compute_food_speed_vs_nb_carriers(True)

# Carrying.compute_nb_carriers_mean_evol(True)
# Carrying.compute_nb_outside_carriers_mean_evol(True)
# Carrying.compute_nb_carriers_mean_evol_around_first_attachment(True)
#
# Carrying.compute_nb_attachments_evol(True)
# Carrying.compute_nb_outside_attachments_evol(True)
# Carrying.compute_nb_attachments_evol_around_first_attachment(True)
# Carrying.compute_nb_outside_attachments_evol_around_first_attachment(True)
# Carrying.compute_nb_non_outside_attachments_evol_around_first_attachment(True)
# Carrying.compute_ratio_outside_attachments_evol_around_first_attachment(True)

FoodVelocity = AnalyseFoodVelocity(group)
# FoodVelocity.compute_food_velocity(True)
# FoodVelocity.compute_food_velocity_phi_hist_evol(True)
# FoodVelocity.compute_food_velocity_phi_hist_evol_around_first_outside_attachment(True)
# FoodVelocity.compute_food_velocity_phi_variance_evol()
# FoodVelocity.compute_food_velocity_phi_variance_evol_around_first_outside_attachment()
#
# FoodVelocity.compute_mm1s_food_velocity_phi()

# FoodVelocity.compute_mm10_food_velocity_vector()
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

AntFoodRelation = AnalyseAntFoodRelation(group)
# AntFoodRelation.compute_ant_density_around_food_evol(True)
# AntFoodRelation.compute_ant_density_around_food_evol_first_outside_attachment(True)
# AntFoodRelation.compute_outside_ant_density_around_food_evol_first_outside_attachment(True)
# AntFoodRelation.compute_non_outside_ant_density_around_food_evol_first_outside_attachment(True)
# AntFoodRelation.compute_slowing_ant_density_around_food_evol_first_outside_attachment(True)
# AntFoodRelation.compute_slowing_outside_ant_density_around_food_evol_first_outside_attachment(True)
# AntFoodRelation.compute_slowing_non_outside_ant_density_around_food_evol_first_outside_attachment(True)
#
# AntFoodRelation.loop_density_around_food_evol(True)
# AntFoodRelation.loop_density_around_food_evol_first_outside_attachment(True)
#
# AntFoodRelation.compute_foodVelocity_foodAntVector_angle()
# AntFoodRelation.compute_foodVelocity_foodAntVector_angle_around_attachments(True)
# AntFoodRelation.compute_foodVelocity_foodAntVector_angle_around_outside_attachments(True)

FoodConfidence = AnalyseFoodConfidence(group)
# FoodConfidence.compute_w2s_food_crossed_distance(True)
# FoodConfidence.compute_w10s_food_crossed_distance(True)
# FoodConfidence.compute_w30s_food_crossed_distance(True)
#
# FoodConfidence.compute_w2s_food_total_crossed_distance(True)
# FoodConfidence.compute_w10s_food_total_crossed_distance(True)
# FoodConfidence.compute_w30s_food_total_crossed_distance(True)
#
# FoodConfidence.compute_w2s_food_path_efficiency(True)
# FoodConfidence.compute_w10s_food_path_efficiency(True)
# FoodConfidence.compute_w30s_food_path_efficiency(True)

FoodVeracity = AnalyseFoodVeracity(group)
# FoodVeracity.compute_food_direction_error(True)

# FoodVeracity.compute_food_direction_error_hist_evol(True)
# FoodVeracity.compute_food_direction_error_variance_evol()

# FoodVeracity.compute_food_direction_error_hist_evol_around_first_attachment(True)
# FoodVeracity.compute_food_direction_error_variance_evol_around_first_attachment(True)

# FoodVeracity.compute_fisher_info_evol_around_first_attachment(True)
#
# FoodVeracity.compute_mm10_food_direction_error(True)
# FoodVeracity.compute_mm1s_food_direction_error(True)
# FoodVeracity.compute_mm10s_food_direction_error(True)

# FoodVeracity.veracity_over_derivative()

FoodInfo = AnalyseFoodInformation(group)
# FoodInfo.compute_w30s_entropy_mm1s_food_velocity_phi_indiv_evol(True)
# FoodInfo.compute_w1s_entropy_mm1s_food_velocity_phi_indiv_evol(True)

# FoodInfo.compute_mm1s_food_direction_error_around_outside_attachments()
# FoodInfo.compute_mm1s_food_direction_error_around_isolated_outside_attachments()
# FoodInfo.compute_mm1s_food_direction_error_around_isolated_non_outside_attachments()
# FoodInfo.compute_mm1s_food_direction_error_around_non_outside_attachments()
# FoodInfo.compute_mm1s_food_direction_error_around_non_outside_attachments_after_first_outside_attachment()
# FoodInfo.compute_mm1s_food_direction_error_around_attachments_after_first_outside_attachment()
# FoodInfo.compute_mm1s_food_direction_error_around_outside_manual_leader_attachments()
# FoodInfo.compute_mm1s_food_direction_error_around_inside_manual_leader_attachments()
# FoodInfo.compute_mm1s_food_direction_error_around_outside_manual_follower_attachments()
# FoodInfo.compute_mm1s_food_direction_error_around_inside_manual_follower_attachments()
#
# FoodInfo.compute_information_mm1s_food_direction_error_around_outside_attachments(True)
# FoodInfo.compute_information_mm1s_food_direction_error_around_isolated_outside_attachments(True)
# FoodInfo.compute_information_mm1s_food_direction_error_around_isolated_non_outside_attachments(True)
# FoodInfo.compute_information_mm1s_food_direction_error_around_non_outside_attachments(True)
# FoodInfo.compute_information_mm1s_food_direction_error_around_attachments_after_first_outside_attachment(True)
# FoodInfo.\
#     compute_information_mm1s_food_direction_error_around_non_outside_attachments_after_first_outside_attachment(True)
# FoodInfo.compute_information_mm1s_food_direction_error_around_attachments(True)
# FoodInfo.compute_information_mm1s_food_direction_error_around_outside_manual_leading_attachments()
# FoodInfo.compute_information_mm1s_food_direction_error_around_inside_manual_leading_attachments()
# FoodInfo.compute_information_mm1s_food_direction_error_around_outside_manual_following_attachments()
# FoodInfo.compute_information_mm1s_food_direction_error_around_inside_manual_following_attachments()

# FoodInfo.compute_information_mm1s_food_direction_error_around_first_outside_attachments(True)
# FoodInfo.compute_information_mm1s_food_direction_error_around_the_first_outside_attachment(True)

# FoodInfo.compute_information_mm1s_food_direction_error_around_outside_attachments_evol(True)
# FoodInfo.compute_information_mm1s_food_direction_error_around_non_outside_attachments_evol(True)
# FoodInfo.\
#     compute_information_mm1s_food_direction_error_around_non_outside_attachments_evol_after_first_outside_attachment(
#         True)
# FoodInfo.compute_information_mm1s_food_direction_error_around_attachments_evol(True)
# #
# FoodInfo.compute_information_mm1s_food_direction_error_over_nbr_new_attachments(True)
# FoodInfo.compute_information_mm1s_food_direction_error_over_nbr_new_outside_attachments(True)
# FoodInfo.compute_information_mm1s_food_direction_error_over_nbr_new_non_outside_attachments(True)
# FoodInfo.compute_information_mm1s_food_direction_error_over_ratio_new_attachments(True)
# FoodInfo.compute_information_mm1s_food_direction_error_around_outside_manual_leading_attachments_evol(True)
# FoodInfo.compute_information_mm1s_food_direction_error_around_inside_manual_leading_attachments_evol(True)
# FoodInfo.compute_information_mm1s_food_direction_error_around_outside_manual_following_attachments_evol(True)
# FoodInfo.compute_information_mm1s_food_direction_error_around_inside_manual_following_attachments_evol(True)
# #
# FoodInfo.compute_fisher_information_mm1s_food_direction_error_around_outside_attachments(True)
# FoodInfo.compute_fisher_information_mm1s_food_direction_error_around_isolated_outside_attachments(True)
# FoodInfo.compute_fisher_information_mm1s_food_direction_error_around_isolated_non_outside_attachments(True)
# FoodInfo.compute_fisher_information_mm1s_food_direction_error_around_non_outside_attachments(True)
# FoodInfo.compute_fisher_information_mm1s_food_direction_error_around_attachments(True)
# #
# FoodInfo.compute_fisher_information_mm1s_food_direction_error_around_first_outside_attachments(True)
# FoodInfo.compute_fisher_information_mm1s_food_direction_error_around_the_first_outside_attachment(True)
#
# FoodInfo.compute_fisher_information_mm1s_food_direction_error_over_nbr_new_attachments(True)
# FoodInfo.compute_fisher_information_mm1s_food_direction_error_over_nbr_new_outside_attachments(True)
# FoodInfo.compute_fisher_information_mm1s_food_direction_error_over_nbr_new_non_outside_attachments(True)
# FoodInfo.compute_fisher_information_mm1s_food_direction_error_over_ratio_new_attachments(True)

# FoodInfo.compute_foodvelocity_foodantvector_angle_around_outside_manual_leader_attachments()
FoodInfo.compute_information_foodvelocity_foodantvector_angle_around_outside_manual_leading_attachments()
# FoodInfo.compute_foodvelocity_foodantvector_angle_around_inside_manual_leader_attachments()
FoodInfo.compute_information_foodvelocity_foodantvector_angle_around_inside_manual_leading_attachments()
# FoodInfo.compute_foodvelocity_foodantvector_angle_around_outside_manual_follower_attachments()
FoodInfo.compute_information_foodvelocity_foodantvector_angle_around_outside_manual_following_attachments()
# FoodInfo.compute_foodvelocity_foodantvector_angle_around_inside_manual_follower_attachments()
FoodInfo.compute_information_foodvelocity_foodantvector_angle_around_inside_manual_following_attachments()

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
# #
# FoodInfoTraj.w10s_smooth_food_direction_error_vs_path_efficiency_scatter_around_first_outside_attachment(True)

# FoodInfoTraj.w10s_food_direction_error_vs_path_efficiency_velocity()
# FoodInfoTraj.w30s_food_direction_error_vs_path_efficiency_velocity()

# FoodInfoTraj.w10s_food_direction_error_vs_path_efficiency_vector_field()
# FoodInfoTraj.w30s_food_direction_error_vs_path_efficiency_vector_field()

# FoodInfoTraj.w10s_food_direction_error_vs_path_efficiency_probability_matrix(False)
# FoodInfoTraj.w30s_food_direction_error_vs_path_efficiency_probability_matrix(False)

LeaderFollower = AnalyseLeaderFollower(group)
# LeaderFollower.get_manual_leader_follower()
# LeaderFollower.print_manual_leader_stats()
# LeaderFollower.prepare_food_speed_features()
# LeaderFollower.test()
