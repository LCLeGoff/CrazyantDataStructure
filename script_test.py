from AnalyseStarter import AnalyseStarter
from ExperimentGroupBuilder import ExperimentGroupBuilder

root = '/data/Dropbox/POSTDOC/CrazyAnt/Results_python/Data/'
group = 'FMAB'

# AnalyseStarter(root, group).start()

# Chara = Characteristics(root, group)
# print(Chara.get_array('entrance').head())

#
ExperimentBuilder = ExperimentGroupBuilder(root)

FMAB = ExperimentBuilder.build(group)
FMAB.load(['x0', 'y0', 'markings', 'food_radius', 'entrance'])
print(FMAB.x0.description, FMAB.x0.name)
print(FMAB.food_radius.description, FMAB.food_radius.name, FMAB.food_radius.array)
# FMAB.spatial_distribution('markings')


