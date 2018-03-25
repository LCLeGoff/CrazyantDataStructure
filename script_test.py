from AnalyseStarter import AnalyseStarter
from CharacteristicsLoader import CharacteristicsLoader
from ComputeTrajectory import ComputeTrajectory
from EventsLoader import EventsLoader
from ExperimentGroupBuilder import ExperimentGroupBuilder
from NameBuilder import NameBuilder
from TimeSeriesLoader import TimeSeriesLoader

root = '/data/Dropbox/POSTDOC/CrazyAnt/Results_python/Data/'
group = 'FMAB'

AnalyseStarter(root, group).start()

NameBuild = NameBuilder(root, group)
CharaLoad = CharacteristicsLoader(root, group)
for name in ['food_radius', 'entrance1']:
	Chara = CharaLoad.load(NameBuild.build(name))
	print(Chara.array.head())
	Chara.operation(lambda x: x/10.)
	print(Chara.array.head())

TSLoad = TimeSeriesLoader(root, group)
for name in ['x0', 'major_axis_length']:
	TS = TSLoad.load(NameBuild.build(name))
	print(TS.array.head())
	TS.operation(lambda x: x/10.)
	print(TS.array.head())

EventsLoad = EventsLoader(root, group)
for name in ['markings']:
	Ev = EventsLoad.load(NameBuild.build(name))
	print(Ev.array.head())
	Ev.operation(lambda x: x/10.)
	print(Ev.array.head())

# FMAB = ExperimentGroupBuilder(root).build(group)
# FMAB.load(['x0', 'y0', 'mm2px', 'entrance', 'crop_limit_x', 'food_center', 'ref_pts', 'entrance'])
# print(FMAB.x0.description, FMAB.x0.name, type(FMAB.x0.array.index[0][0]))
# print(FMAB.food_center.description, FMAB.food_center.name, type(FMAB.x0.array.index[0][0]))
# print(FMAB.crop_limit_x.description, FMAB.crop_limit_x.name)
# print(FMAB.mm2px.description, FMAB.mm2px.name)
# print(FMAB.ref_pts.description, FMAB.ref_pts.name)
# print(FMAB.entrance.description, FMAB.entrance.name)

# traj = ComputeTrajectory(root, group)
# traj.centered_x_y()
