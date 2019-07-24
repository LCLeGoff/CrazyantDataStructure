import numpy as np

from Scripts.root import root
from AnalyseClasses.Models.UOModels import UOSimpleModel, AnalyseUOModel, UOConfidenceModel, PersistenceModel, UORWModel

group = 'UO'
AnalyseUOModel = AnalyseUOModel(group)


# SimpleModel = UOSimpleModel(root, group, new=True, n_replica=500)
#
# p_attach = 0.6
# a = 0.98
# r = 1.58
# c = round(1-np.sqrt(1-(1-a)/p_attach), 5)
#
# min_var_orientation = (1-a)*r-p_attach*c**2
# max_var_orientation = 0.0316
# var_orientation = np.around(np.mean([min_var_orientation, max_var_orientation]), 5)
# var_info = round(((1-a)*r-var_orientation)/p_attach/c**2, 5)
#
# SimpleModel.run({'c': c, 'p_attachment': p_attach, 'var_orientation': var_orientation, 'var_information': var_info})
# SimpleModel.write()
#
# AnalyseUOModel.plot_simple_model_evol(suff='high_persistence')
# # # AnalyseUOModel.plot_simple_model_evol()

RWModel = UORWModel(root, group, new=True, n_replica=500)

c = 1
p_attach = 0.5
d_orientation = .1
d_info = .1

RWModel.run({'c': c, 'p_attachment': p_attach, 'd_orientation': d_orientation, 'd_information': d_info})
RWModel.write()

AnalyseUOModel.plot_rw_model_evol()


# # # ConfidenceModel = UOConfidenceModel(root, group, new=True, n_replica=250)
# # # var_orientation = 0.1
# # # var_perception = 0.
# # # p_att = 0.5
# # # for var_information in [0.1, 0.5, 1, 2, 3, 5]:
# # #     ConfidenceModel.run({'p_attachment': p_att, 'var_orientation': var_orientation,
# # #                         'var_perception': var_perception, 'var_information': var_information})
# # # ConfidenceModel.write()
# # # AnalyseUOModel.plot_confidence_model_evol(suff='high_persistence')
# #
# # PersistenceModel = PersistenceModel(root, group, new=True, duration=500, n_replica=250)
# # PersistenceModel.run({'var_orientation': 1})
# # PersistenceModel.run({'var_orientation': 0.25})
# # PersistenceModel.run({'var_orientation': 0.09})
# # PersistenceModel.run({'var_orientation': 0.0316})
# # PersistenceModel.run({'var_orientation': 0.01})
# # PersistenceModel.write()
#
# # AnalyseUOModel.plot_persistence()

