import numpy as np

from Scripts.root import root
from AnalyseClasses.Models.UOModels import UOSimpleModel, AnalyseUOModel, UOConfidenceModel, PersistenceModel


group = 'UO'


def get_var_info(c2, p_attach, var_orient):
    k = 1.75
    return np.around((k * p_attach * c2 * (2 - c2) - var_orient) / p_attach / c2 ** 2, 3)


SimpleModel = UOSimpleModel(root, group, new=True, n_replica=500)
var_orientation = 0.09
p_att = 0.1
for c in [0.35, 0.5, 1]:
    var_information = get_var_info(c, p_att, var_orientation)
    if var_information > 0:
        SimpleModel.run({'c': c, 'p_attachment': p_att, 'var_orientation': var_orientation,
                         'var_information': var_information})
SimpleModel.write()

AnalyseUOModel = AnalyseUOModel(group)
# AnalyseUOModel.plot_simple_model_evol(suff='high_persistence')
AnalyseUOModel.plot_simple_model_evol()

ConfidenceModel = UOConfidenceModel(root, group, new=True, n_replica=250)
# var_orientation = 0.1
# var_perception = 0.
# p_att = 0.5
# for var_information in [0.1, 0.5, 1, 2, 3, 5]:
#     ConfidenceModel.run({'p_attachment': p_att, 'var_orientation': var_orientation,
#                         'var_perception': var_perception, 'var_information': var_information})
# ConfidenceModel.write()
#
PersistenceModel = PersistenceModel(root, group, new=True, duration=60, n_replica=1000)
# PersistenceModel.run({'var_orientation': 1})
# PersistenceModel.run({'var_orientation': 0.25})
# PersistenceModel.run({'var_orientation': 0.09})
# PersistenceModel.run({'var_orientation': 0.01})
# PersistenceModel.write()
#
# # AnalyseUOModel.plot_confidence_model_evol(suff='high_persistence')
# AnalyseUOModel.plot_persistence()
