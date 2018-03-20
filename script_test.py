from AnalyseStarter import AnalyseStarter
from Characteristics import Characteristics
from ExperimentGroupBuilder import ExperimentGroupBuilder

root = '/data/Dropbox/POSTDOC/CrazyAnt/Results_python/Data/'
group = 'FMAB'

# AnalyseStarter(root, group).start()

# Chara = Characteristics(root, group)
# print(Chara.get_array('entrance').head())

#
ExperimentBuilder = ExperimentGroupBuilder(root)

FMAB = ExperimentBuilder.build(group)
FMAB.load(['x0', 'y0', 'markings'])
print(FMAB.x0.description)
# FMAB.spatial_distribution('markings')


