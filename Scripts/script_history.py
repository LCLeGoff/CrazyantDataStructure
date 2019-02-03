from AnalyseClasses.Markings.BaseMarkings import AnalyseMarkings
from AnalyseClasses.AnalyseStarter import AnalyseStarter
from AnalyseClasses.Markings.Recruitment import Recruitment
from AnalyseClasses.Markings.RecruitmentDirection import RecruitmentDirection
from AnalyseClasses.Trajectory.BaseTrajectory import AnalyseTrajectory
from Scripts import root

for group in ['FMAB', 'FMABU', 'FMABW']:
    print(group)
    starter = AnalyseStarter(root, group)
    # starter.start(False)

    traj = AnalyseTrajectory(root, group)
    # traj.compute_x_y()
    # traj.compute_r_phi()
    #
    mark = AnalyseMarkings(root, group)
    # mark.compute_xy_marking()
    # mark.compute_r_phi_marking()
    mark.compute_marking_interval()
    # mark.compute_marking_distance()

    recruit = Recruitment(root, group)
    # recruit.compute_marking_batch()
    # recruit.compute_recruitment()

    recruit_direction = RecruitmentDirection(root, group)
    recruit_direction.compute_recruitment_direction()
