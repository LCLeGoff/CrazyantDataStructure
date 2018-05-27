from AnalyseClasses.Markings.Recruitment import Recruitment
from AnalyseClasses.Markings.TestRecruitment import TestRecruitment
from Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from DataManager.Loaders.CharacteristicsLoader import Characteristics1dLoader, Characteristics2dLoader
from DataManager.Loaders.DefinitionLoader import DefinitionLoader
from DataManager.Loaders.EventsLoader import Events1dLoader
from DataManager.Loaders.TimeSeriesLoader import TimeSeries1dLoader
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

# for group in ['FMABU']:
# 	print(group)
# 	test_recruit = TestRecruitment(root, group)
# 	# test_recruit.compute_first_marking_ant_radial_criterion(show=False)
# 	test_recruit.compute_first_marking_ant_batch_criterion(show=True, id_exp_list=range(8, 9))
# 	# test_recruit.compute_first_marking_ant_setup_orientation()
# 	# test_recruit.compute_first_marking_ant_setup_orientation_circle()
# plt.show()
# #
# #
for group in ['FMAB']:
	print(group)
	recruit = Recruitment(root, group)
	# recruit.compute_marking_batch()
	recruit.compute_recruitment(range(10))
plt.show()
