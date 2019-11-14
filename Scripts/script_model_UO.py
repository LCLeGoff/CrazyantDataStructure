import numpy as np

from AnalyseClasses.Models.UOModelAnalysis import UOModelAnalysis
from Scripts.root import root
from AnalyseClasses.Models.UOModels import UOSimpleModel, PlotUOModel, UOConfidenceModel, PersistenceModel, \
    UORWModel, UOOutsideModel

group = 'UO'
PlotUOModel = PlotUOModel(group)


SimpleModel = UOSimpleModel(root, group, new=True, n_replica=10000)
#
# p_attach = 0.65
# a = 0.98
# r = 1.72
# c = round(1-np.sqrt(1-(1-a)/p_attach), 3)
# print((1-a)*r)
# # #
# # # min_var_orientation = (1-a)*r-p_attach*c**2
# # # max_var_orientation = 0.031
# # # var_orientation = np.around(np.mean([min_var_orientation, max_var_orientation]), 5)
# var_orientation = np.around((1-a)*r-p_attach*c**2, 3)
# var_info = 1
#
# SimpleModel.run({'c': c, 'p_attachment': p_attach, 'var_orientation': var_orientation, 'var_information': 0})
# SimpleModel.write()


# PlotUOModel.plot_simple_model_evol(suff='high_persistence')
# PlotUOModel.plot_indiv_hist_evol('UOSimpleModel', 0)
# # PlotUOModel.plot_simple_model_evol()


group = 'UO'
ModelAnalysis = UOModelAnalysis(group)
ModelAnalysis.compute_model_attachment_intervals('UOSimpleModel')

# OutsideModel = UOOutsideModel(root, group, new=True, n_replica=1000, time0=0)
#
# var_info = 1.
# var_orientation = 0.09
# for c in [0.01, 0.02, 0.05, 0.075, 0.1, 0.2]:
#     OutsideModel.run(
#         {'c': c, 'var_orientation': var_orientation, 'var_information': var_info})
#
#
# OutsideModel.write()
#
# PlotUOModel.plot_outside_model_evol()
# PlotUOModel.plot_hist_model_pretty('UOOutsideModel')
# PlotUOModel.plot_var_model_pretty('UOOutsideModel')
# PlotUOModel.plot_hist_model_pretty('UOSimpleModel', display_title=False)
# PlotUOModel.plot_var_model_pretty('UOSimpleModel', display_title=False)

# RWModel = UORWModel(root, group, new=True, n_replica=500)

# c = 1
# p_attach = 0.5
# d_orientation = .1
# d_info = .1
#
# RWModel.run({'c': c, 'p_attachment': p_attach, 'd_orientation': d_orientation, 'd_information': d_info})
# RWModel.write()

# PlotUOModel.plot_rw_model_evol()


# # # ConfidenceModel = UOConfidenceModel(root, group, new=True, n_replica=250)
# # # var_orientation = 0.1
# # # var_perception = 0.
# # # p_att = 0.5
# # # for var_information in [0.1, 0.5, 1, 2, 3, 5]:
# # #     ConfidenceModel.run({'p_attachment': p_att, 'var_orientation': var_orientation,
# # #                         'var_perception': var_perception, 'var_information': var_information})
# # # ConfidenceModel.write()
# # # PlotUOModel.plot_confidence_model_evol(suff='high_persistence')
# #
# # PersistenceModel = PersistenceModel(root, group, new=True, duration=500, n_replica=250)
# # PersistenceModel.run({'var_orientation': 1})
# # PersistenceModel.run({'var_orientation': 0.25})
# # PersistenceModel.run({'var_orientation': 0.09})
# # PersistenceModel.run({'var_orientation': 0.0316})
# # PersistenceModel.run({'var_orientation': 0.01})
# # PersistenceModel.write()
#
# # PlotUOModel.plot_persistence()

