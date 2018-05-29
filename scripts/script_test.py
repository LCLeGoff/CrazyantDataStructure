from AnalyseClasses.Markings.Recruitment import Recruitment
from AnalyseClasses.Markings.RecruitmentDirection import RecruitmentDirection
from AnalyseClasses.Markings.TestRecruitment import TestRecruitment
from Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from DataManager.Loaders.CharacteristicsLoader import Characteristics1dLoader, Characteristics2dLoader
from DataManager.Loaders.DefinitionLoader import DefinitionLoader
from DataManager.Loaders.EventsLoader import Events1dLoader
from DataManager.Loaders.TimeSeriesLoader import TimeSeries1dLoader
from Plotter.Plotter2d import Plotter2d
from Tools.Geometry import distance
from scripts.root import root
from AnalyseClasses.Markings.BaseMarkings import AnalyseMarkings
from matplotlib import pyplot as plt
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
# 	# recruit.compute_marking_batch()
# 	recruit.compute_recruitment()
# plt.show()

# for group in ['FMAB', 'FMABU', 'FMABW']:
# 	exp = ExperimentGroupBuilder(root).build(group)
# 	exp.rename_data('marking_batch_interval', 'marking_batch_intervals')
# 	exp.rename_data('marking_interval', 'marking_intervals')
# 	exp.rename_data('marking_batch_time_threshold', 'marking_batch_time_thresholds')
# 	exp.rename_data('marking_batch_distance_threshold', 'marking_batch_distance_thresholds')
# 	exp.rename_data('polar_markings', 'rphi_markings')
# 	exp.rename_data('marking_distance', 'marking_distances')


# for group in ['FMAB']:
# 	print(group)
# 	recruit_direction = RecruitmentDirection(root, group)
# 	recruit_direction.compute_recruitment_direction()

# plot2d = Plotter2d()
# fig, ax = plot2d.create_plot()
# plot2d.draw_setup(fig, ax)
# plt.show()

for group in ['FMAB', 'FMABU', 'FMABW']:
	exp = ExperimentGroupBuilder(root).build(group)
	# preplot = Plotter2d().create_plot(figsize=(6.5, 5))
	# exp.load_as_2d('x', 'y', 'xy')
	# preplot = exp.xy.plotter.repartition_in_arena(ls='-', lw=1, marker='')
	# exp.load('marking_intervals')
	# exp.filter_with_time_intervals('xy', 'marking_intervals', 'marking_traj')
	# exp.marking_traj.plot_repartition_in_arena(preplot=preplot, ls='-', lw=1, marker='')
	# exp.load(['xy_markings', 'recruitment_intervals'])
	# preplot = exp.xy_markings.plotter.repartition_in_arena(alpha=1)
	# exp.filter_with_time_intervals(
	# 	'xy_markings', 'recruitment_intervals', 'xy_recruitments', label='', xlabel='', ylabel='', replace=True)
	# exp.load('xy_recruitments')
	# exp.xy_recruitments.plotter.repartition_in_arena(color_variety='frame', preplot=preplot, title_prefix=group, alpha=1)
	exp.load(['phi_markings', 'recruitment_intervals'])
	dtheta = 0.2
	bins = np.arange(0, np.pi+dtheta/2., dtheta)
	exp.operation('phi_markings', lambda x: np.abs(x))
	preplot = exp.phi_markings.plotter.hist1d(c='r', bins=bins, normed=True, xscale='log', yscale='log')
	exp.filter_with_time_intervals(
		'phi_markings', 'recruitment_intervals', 'phi_recruitments', label='', xlabel='', ylabel='', replace=True)
	exp.phi_recruitments.plotter.hist1d(
		bins=bins, normed=True, title_prefix=group, preplot=preplot, xscale='log', yscale='log')

plt.show()
