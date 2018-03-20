from AnalyseStarter import AnalyseStarter
from ExperimentGroupBuilder import ExperimentGroupBuilder

root = '/data/Dropbox/POSTDOC/CrazyAnt/Results_python/Data/'
group = 'FMAB'

# AnalyseStarter(root, group).start()

ExperimentBuilder = ExperimentGroupBuilder(root)

FMAB = ExperimentBuilder.build(group)

FMAB.load(['x', 'y', 'markings'])
