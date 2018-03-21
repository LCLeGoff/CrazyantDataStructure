from AnalyseStarter import AnalyseStarter
from ExperimentGroupBuilder import ExperimentGroupBuilder

root = '/data/Dropbox/POSTDOC/CrazyAnt/Results_python/Data/'
group = 'FMAB'

# AnalyseStarter(root, group).start()

# Chara = Characteristics(root, group)
# print(Chara.get_array('entrance').head())


FMAB = ExperimentGroupBuilder(root).build(group)
FMAB.load(['x0', 'y0', 'mm2px', 'entrance', 'crop_limit_x', 'food_center', 'ref_pts', 'entrance'])

print(FMAB.x0.description, FMAB.x0.name, type(FMAB.x0.array.index[0][0]))
print(FMAB.food_center.description, FMAB.food_center.name, type(FMAB.x0.array.index[0][0]))
print(FMAB.crop_limit_x.description, FMAB.crop_limit_x.name)
print(FMAB.mm2px.description, FMAB.mm2px.name)
print(FMAB.ref_pts.description, FMAB.ref_pts.name)
print(FMAB.entrance.description, FMAB.entrance.name)


