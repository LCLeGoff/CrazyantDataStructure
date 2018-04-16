
import numpy as np
from AnalyseClasses.AnalyseMarkings import AnalyseMarkings
from AnalyseClasses.AnalyseTrajectory import AnalyseTrajectory
from matplotlib import pyplot as plt

from AnalyseStarter import AnalyseStarter
from Builders.ExperimentGroupBuilder import ExperimentGroupBuilder

root = '/data/Dropbox/POSTDOC/CrazyAnt/Results_python/Data/'
# group = 'FMABW'

# for group in ['FMAB', 'FMABU', 'FMABW']:
# 	AnalyseStarter(root, group).start(False)

# DefinitionBuilder = DefinitionBuilder(root, group)
#
# CharaLoad = Characteristics1dLoader(root, group)
# for name in ['food_radius', 'mm2px']:
# 	Chara = CharaLoad.load(DefinitionBuilder.build(name))
# 	Chara.print()
# 	Chara.operation(lambda x: x/10.)
# 	Chara.print()
#
# CharaLoad = Characteristics2dLoader(root, group)
# for name in ['ref_pts1', 'entrance1']:
# 	Chara = CharaLoad.load(DefinitionBuilder.build(name))
# 	Chara.print()
# 	Chara.operation(lambda x: x/10.)
# 	Chara.print()

# TSLoad = TimeSeriesLoader(root, group)
# for name in ['x0', 'major_axis_length']:
# 	TS = TSLoad.load(NameBuild.build(name))
# 	print(TS.array.head())
# 	TS.operation(lambda x: x/10.)
# 	print(TS.array.head())
#
# EventsLoad = EventsLoader(root, group)
# for name in ['markings']:
# 	Ev = EventsLoad.load(NameBuild.build(name))
# 	print(Ev.array.head())
# 	Ev.operation(lambda x: x/10.)
# 	print(Ev.array.head())

# FMAB = ExperimentGroupBuilder(root).build(group)
# FMAB.load(['x0', 'y0', 'mm2px', 'crop_limit_x', 'food_center', 'ref_pts2', 'entrance1'])
# print(FMAB.x0.description, FMAB.x0.name, type(FMAB.x0.array.index[0][0]))
# print(FMAB.food_center.description, FMAB.food_center.name, type(FMAB.x0.array.index[0][0]))
# print(FMAB.crop_limit_x.description, FMAB.crop_limit_x.name)
# print(FMAB.mm2px.description, FMAB.mm2px.name)
# print(FMAB.ref_pts2.description, FMAB.ref_pts2.name)
# print(FMAB.entrance0.description, FMAB.entrance0.name)

# for group in ['FMAB', 'FMABU', 'FMABW']:
# 	traj = AnalyseTrajectory(root, group)
# 	traj.centered_x_y()
# 	traj.xy_polar()

for group in ['FMAB']:
	mark = AnalyseMarkings(root, group)
	# mark.compute_xy_marking()
	# mark.compute_xy_marking_polar()
	# mark.marking_interval()
	# exp = ExperimentGroupBuilder(root).build(group)
	# exp.load('marking_interval')
	# exp.plot_hist1d('marking_interval', range(0, 1000), xscale='log', yscale='log', group=0)
	# mark.compute_first_marking_ant_radial_criterion(show=False)
	mark.compute_first_marking_ant_batch_criterion(show=True)
	# mark.compute_first_marking_ant_setup_orientation()
	# mark.compute_first_marking_ant_setup_orientation_circle()
plt.show()
#

# for group in ['FMAB', 'FMABU', 'FMABW']:
# 	mark = AnalyseMarkings(root, group)
# 	# mark.spatial_repartition_xy_markings()
# 	# mark.spatial_repartition_xy_markings_2d_hist()
# 	# mark.spatial_repartition_first_markings()
#
# 	# mark.spatial_repartition_first_markings_2d_hist()
# plt.show()
