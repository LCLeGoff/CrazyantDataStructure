from ExperimentGroupBuilder import ExperimentGroupBuilder

root = '/data/Dropbox/POSTDOC/CrazyAnt/Results_python/Data/'
group = 'FMAB'
ExperimentBuilder = ExperimentGroupBuilder(root)


id_exp = 1
exp = ExperimentBuilderFMAB.build(id_exp)
print(exp)
