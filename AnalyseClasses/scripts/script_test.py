from AnalyseClasses.Markings.RecruitmentDirection import RecruitmentDirection
from AnalyseClasses.scripts.root import root
from matplotlib import pyplot as plt

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Tools.Plotter.BasePlotters import BasePlotters
import numpy as np

group = 'FMAB'

# DefinitionLoader = DefinitionLoader(root, group)

# CharaLoad = Characteristics1dLoader(root, group)
# for name in ['food_radius', 'mm2px']:
# 	print(name)
# 	Chara = CharaLoad.load(DefinitionLoader.build(name))
# 	Chara.print()
# 	Chara.operation(lambda x: x/10.)
# 	Chara.print()
#
# CharaLoad = Characteristics2dLoader(root, group)
# for name in ['ref_pts1', 'entrance1']:
# 	print(name)
# 	Chara = CharaLoad.load(DefinitionLoader.build(name))
# 	Chara.print()
# 	Chara.operation(lambda x: x/10.)
# 	Chara.print()
# #
# TSLoad = TimeSeriesLoader(root, group)
# for name in ['x0', 'major_axis_length']:
# 	print(name)
# 	TS = TSLoad.load(DefinitionLoader.build(name))
# 	print(TS.df.head())
# 	TS.operation(lambda x: x/10.)
# 	print(TS.df.head())
#
# EventsLoad = EventsLoader(root, group)
# for name in ['markings']:
# 	print(name)
# 	Ev = EventsLoad.load(DefinitionLoader.build(name))
# 	print(Ev.df.head())
# 	Ev.operation(lambda x: x/10.)
# 	print(Ev.df.head())

# FMAB = ExperimentGroupBuilder(root).build(group)
# FMAB.load(['x0', 'y0', 'mm2px', 'crop_limit_x', 'food_center', 'ref_pts2', 'entrance1'])
# print(FMAB.x0.description, FMAB.x0.name, FMAB.x0.df.index[0][0])
# print(FMAB.food_center.description, FMAB.food_center.name, type(FMAB.x0.df.index[0][0]))
# print(FMAB.crop_limit_x.description, FMAB.crop_limit_x.name)
# print(FMAB.mm2px.description, FMAB.mm2px.name)
# print(FMAB.ref_pts2.description, FMAB.ref_pts2.name)
# print(FMAB.entrance1.description, FMAB.entrance0.name)

#
# for group in ['FMAB', 'FMABU', 'FMABW']:
# 	print(group)
# 	exp = ExperimentGroupBuilder(root).build(group)
# 	exp.load('marking_interval')
# 	exp.plot_hist1d('marking_interval', 'fd', xscale='log', yscale='log', group=0)
#
# 	exp.load('xy_markings')
# 	array_of_idx_exp_ant = exp.xy_markings.get_array_id_exp_ant()
# 	dist = []
# 	for (id_exp, id_ant) in array_of_idx_exp_ant:
# 		xy_marks = exp.xy_markings.get_row_id_exp_ant(id_exp, id_ant)
# 		xy = np.array(xy_marks)
# 		dist += list(distance(xy[1:], xy[:-1]))
# 	y, x = np.histogram(dist, 'fd')
# 	x = (x[1:] + x[:-1]) / 2.
# 	plt.loglog(x, y, '.-')
# plt.show()


# for group in ['FMAB', 'FMABU', 'FMABW']:
# 	mark = AnalyseMarkings(root, group)
# 	# mark.spatial_repartition_xy_markings()
# 	# mark.spatial_repartition_xy_markings_2d_hist()
# 	# mark.spatial_repartition_first_markings()
#
# 	# mark.spatial_repartition_first_markings_2d_hist()
# plt.show()

# for group in ['FMABW']:
# 	print(group)
# 	test_recruit = TestRecruitment(root, group)
# 	# test_recruit.compute_first_marking_ant_radial_criterion(show=False)
# 	test_recruit.compute_first_marking_ant_batch_criterion(show=True)
# 	# test_recruit.compute_first_marking_ant_setup_orientation()
# 	# test_recruit.compute_first_marking_ant_setup_orientation_circle()
# plt.show()
# #
# #
# for group in ['FMAB']:
# 	print(group)
# 	recruit = Recruitment(root, group)
# 	recruit.compute_marking_batch()
# 	# recruit.compute_recruitment()
# plt.show()

# for group in ['FMAB', 'FMABU', 'FMABW']:
#     exp = ExperimentGroupBuilder(root).build(group)
#     # exp.rename_data('marking_batch_interval', 'marking_batch_intervals')
#     # exp.rename_data('marking_interval', 'marking_intervals')
#     # exp.rename_data('marking_batch_time_threshold', 'marking_batch_time_thresholds')
#     # exp.rename_data('marking_batch_distance_threshold', 'marking_batch_distance_thresholds')
#     # exp.rename_data('polar_markings', 'rphi_markings')
#     # exp.rename_data('marking_distance', 'marking_distances')
#     # exp.rename_data('ab_recruitment', 'ab_recruitments')
#     # exp.rename_data('ab_recruitment_orientS', 'ab_recruitments_orientS')
#     # exp.rename_data('recruitment_direction', 'recruitment_directions')
#     # exp.rename_data('recruitment_direction_orientS', 'recruitment_directions_orientS')


# plot2d = Plotter2d()
# fig, ax = plot2d.create_plot()
# plot2d.draw_setup(fig, ax)
# plt.show()

# for group in ['FMAB', 'FMABU', 'FMABW']:
# exp = ExperimentGroupBuilder(root).build(group)
# # preplot = Plotter2d().create_plot(figsize=(6.5, 5))
# # exp.load_as_2d('x', 'y', 'xy')
# # preplot = exp.xy.plotter.repartition_in_arena(ls='-', lw=1, marker='')
# # exp.load('marking_intervals')
# # exp.filter_with_time_intervals('xy', 'marking_intervals', 'marking_traj')
# # exp.marking_traj.plot_repartition_in_arena(preplot=preplot, ls='-', lw=1, marker='')
# # exp.load(['xy_markings', 'recruitment_intervals'])
# # preplot = exp.xy_markings.plotter.repartition_in_arena(alpha=1)
# # exp.filter_with_time_intervals(
# # 	'xy_markings', 'recruitment_intervals', 'xy_recruitments', label='', xlabel='', ylabel='', replace=True)
# # exp.load('xy_recruitments')
# # exp.xy_recruitments.plotter.repartition_in_arena(
# #     color_variety='frame', preplot=preplot, title_prefix=group, alpha=1)
# exp.load(['phi_markings', 'recruitment_intervals'])
# dtheta = 0.2
# bins = np.arange(0, np.pi+dtheta/2., dtheta)
# exp.operation('phi_markings', lambda x: np.abs(x))
# preplot = exp.phi_markings.plotter.hist1d(c='r', bins=bins, normed=True, xscale='log', yscale='log')
# exp.filter_with_time_intervals(
# 	'phi_markings', 'recruitment_intervals', 'phi_recruitments', label='', xlabel='', ylabel='', replace=True)
# exp.phi_recruitments.plotter.hist1d(
# 	bins=bins, normed=True, title_prefix=group, preplot=preplot, xscale='log', yscale='log')

# plt.show()
#
# for i, group in enumerate(['FMAB']):
#     print(group)
#     recruit_direction = RecruitmentDirection(root, group)
#     # recruit_direction.compute_recruitment_direction()
#
#     recruit_direction.exp.load([
#         'xy_recruitments',
#         'xy_recruitments_in_circular_arena_orientS',
#         'recruitment_directions_orientS',
#         'ab_recruitment_orientS'
#     ])
#     first_recruitment_direction_name = 'first_recruitment_directions'
#
#     recruit_direction.exp.load_as_2d('x', 'y', 'xy')
#     for id_exp in recruit_direction.exp.id_exp_list:
#         preplot = BasePlotters().create_plot(figsize=(13, 10))
#         recruit_direction.exp.xy_recruitments_in_circular_arena_orientS.plotter.repartition_in_arena(
#             preplot=preplot, list_id_exp=[id_exp], color_variety='ant2', title_prefix=str(id_exp))
#         # recruit_direction.exp.recruitment_directions_orientS.plotter.radial_direction_in_arena(
#         #     center_obj=recruit_direction.exp.xy,
#         #     preplot=preplot, list_id_exp=[id_exp], color_variety='ant2', ls='-')
#         recruit_direction.exp.ab_recruitment_orientS.plotter.plot_ab_line(
#             preplot=preplot, list_id_exp=[id_exp], color_variety='ant2', ls='-')
#         plt.show()


col = ['w', 'y', 'r']
labels = ['nothing', 'sight only', 'smell and sight']
labels2 = ['all recruitment', 'first recruitment']
dt = 1 / 5.
preplot1 = BasePlotters().create_plot(figsize=(6.5, 5))
# preplot2 = BasePlotters().create_plot(figsize=(6.5, 5))
for i, group in enumerate(['FMAB', 'FMABU', 'FMABW']):
    # preplot = BasePlotters().create_plot(figsize=(6.5, 5))
    print(group)
    recruit_direction = RecruitmentDirection(root, group)
    recruit_direction.exp.load([
        'recruitment_directions_orientS', 'first_recruitment_directions_orientS',
    ])

    # recruit_direction.exp.recruitment_directions_orientS.plotter.hist1d(
    #     title_prefix=labels[i], bins=np.arange(-1 + dt / 2., 1 + dt / 2., dt) * np.pi,
    #     marker='o', lw=1, normed=True, preplot=preplot, c='b', label='all recruitments')
    # recruit_direction.exp.first_recruitment_directions_orientS.plotter.hist1d(
    #     title_prefix=labels[i] + ' (first)', bins=np.arange(-1 + dt / 2., 1 + dt / 2., dt) * np.pi,
    #     marker='o', lw=1, normed=True, preplot=preplot, c='r', label='first recruitment')
#
    recruit_direction.exp.recruitment_directions_orientS.plotter.hist1d(
        title_prefix=labels[i], bins=np.arange(-1 + dt / 2., 1 + dt / 2., dt) * np.pi,
        marker='o', lw=1, normed=True, preplot=preplot1, c=col[i], label=labels[i])
    # recruit_direction.exp.first_recruitment_directions_orientS.plotter.hist1d(
    #     title_prefix=labels[i] + ' (first)', bins=np.arange(-1 + dt / 2., 1 + dt / 2., dt) * np.pi,
    #     marker='o', lw=1, normed=True, preplot=preplot2, c=col[i], label=labels[i])
    # preplot[0].legend()
#
preplot1[0].legend()
# preplot2[0].legend()
plt.show()


col = ['w', 'y', 'r']
labels = ['nothing', 'smell only', 'smell and sight']
labels2 = ['all recruitment', 'first recruitment']
dt = 1 / 5.
preplot1 = BasePlotters().create_plot(figsize=(6.5, 5))
preplot2 = BasePlotters().create_plot(figsize=(6.5, 5))
for i, group in enumerate(['FMAB', 'FMABU', 'FMABW']):
    # preplot = BasePlotters().create_plot(figsize=(6.5, 5))
    print(group)
    recruit_direction = RecruitmentDirection(root, group)
    recruit_direction.exp.load([
        'recruitment_certainty_orientS', 'recruitment_directions_orientS',
        'first_recruitment_certainty_orientS', 'first_recruitment_directions_orientS'
    ])
    # recruit_direction.exp.recruitment_certainty_orientS.plotter.hist1d(
    #     title_prefix=labels[i],
    #     # yscale='log',
    #     # bins=np.arange(-1 + dt / 2., 1 + dt / 2., dt) * np.pi,
    #     marker='o', lw=1, normed=True, preplot=preplot1, c=col[i], label=labels[i])
    #
    # recruit_direction.exp.add_2d_from_1ds(
    #     name1='recruitment_directions_orientS', name2='recruitment_certainty_orientS',
    #     result_name='certainty_vs_direction'
    # )
    #
    # recruit_direction.exp.certainty_vs_direction.plotter.plot_scatter(
    #     preplot=preplot2, c=col[i], title_prefix=labels[i], label=labels[i])

    recruit_direction.exp.first_recruitment_certainty_orientS.plotter.hist1d(
        title_prefix=labels[i],
        # yscale='log',
        # bins=np.arange(-1 + dt / 2., 1 + dt / 2., dt) * np.pi,
        marker='o', lw=1, normed=True, preplot=preplot1, c=col[i], label=labels[i])

    recruit_direction.exp.add_2d_from_1ds(
        name1='first_recruitment_directions_orientS', name2='first_recruitment_certainty_orientS',
        result_name='certainty_vs_direction'
    )

    recruit_direction.exp.certainty_vs_direction.plotter.plot_scatter(
        preplot=preplot2, c=col[i], title_prefix=labels[i], label=labels[i])


preplot1[0].legend()
# preplot2[0].legend()
plt.show()
