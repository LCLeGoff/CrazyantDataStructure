import numpy as np

from AnalyseClasses.Models.UOModelAnalysis import UOModelAnalysis
from Scripts.root import root
from AnalyseClasses.Models.UOModels import UOSimpleModel, PlotUOModel, UOConfidenceModel, PersistenceModel, \
    UORWModel, UOOutsideModel, UORandomcSimpleModel, UOCSimpleModel, UniformPersistenceModel, UOUniformSimpleModel, \
    UOWrappedSimpleModel, WrappedPersistenceModel

group = 'UO'
PlotUOModel = PlotUOModel(group)

#
# SimpleModel = UOSimpleModel(root, group, new=True, n_replica=10000)
# #
# p_attach = 0.051
# a = np.exp(-0.0283)
# r = 1.7
# var_orientation = 0.04
#
# c = round(1-np.sqrt(1-(1-a)/p_attach), 3)
# var_info = (r*p_attach*c*(2-c)-(1-p_attach)*var_orientation)/(p_attach*c**2)
# SimpleModel.run({'c': c, 'p_attachment': p_attach, 'var_orientation': var_orientation, 'var_information': var_info})
#
# # c = 1.
# # SimpleModel.run({'c': c, 'p_attachment': p_attach, 'var_orientation': var_orientation, 'var_information': var_info})
#
# c = 0.5
# SimpleModel.run({'c': c, 'p_attachment': p_attach, 'var_orientation': var_orientation, 'var_information': var_info})
# #
# # var_orientation = 0.09
# # # var_info = r*(r*p_attach/((1-p_attach)*var_orientation)-1)
# # # c = r/(r+var_info)
# # # SimpleModel.run({'c': c, 'p_attachment': p_attach, 'var_orientation': var_orientation, 'var_information': var_info})
# # # #
# # # # var_orientation = 0.09
# # # # c = 0.9
# # # # var_info = (r*p_attach*c*(2-c)-(1-p_attach)*var_orientation)/(p_attach*c**2)
# # # # SimpleModel.run({'c': c, 'p_attachment': p_attach, 'var_orientation': var_orientation, 'var_information': var_info})
# # #
# SimpleModel.write()
# #
# PlotUOModel.plot_hist_model_pretty('UOSimpleModel', display_title=False)
# PlotUOModel.plot_var_model_pretty('UOSimpleModel', display_title=False)


# PlotUOModel.plot_simple_model_evol()
# PlotUOModel.plot_indiv_hist_evol('UOSimpleModel', 0)
# # PlotUOModel.plot_simple_model_evol()

# WrappedSimpleModel = UOWrappedSimpleModel(root, group, new=True, n_replica=10000)
#
# p_attach = 0.051
# kappa_orientation = 15
#
# kappa_info = 2.38
# c = 1-2*np.pi*0.15/2.
# WrappedSimpleModel.run(
#     {'c': c, 'p_attachment': p_attach, 'kappa_orientation': kappa_orientation, 'kappa_information': kappa_info})
#
# kappa_info = 4
# c = 1-2*np.pi*0.15/2.
# WrappedSimpleModel.run(
#     {'c': c, 'p_attachment': p_attach, 'kappa_orientation': kappa_orientation, 'kappa_information': kappa_info})
#
# WrappedSimpleModel.write()
# PlotUOModel.exp = WrappedSimpleModel.exp
# PlotUOModel.plot_hist_model_pretty('UOWrappedSimpleModel', display_title=False)
# PlotUOModel.plot_var_model_pretty('UOWrappedSimpleModel', display_title=False)

# RandomcModel = UORandomcSimpleModel(root, group, new=True, n_replica=10000)
# p_attach = 0.052
# a = np.exp(-0.0283)
# r = 1.7
# var_orientation = 0.09
# var_info = r*(r*p_attach/((1-p_attach)*var_orientation)-1)
# c = r/(r+var_info)
# var_c = 0.1
# RandomcModel.run(
#     {'c': c, 'var_c': var_c, 'p_attachment': p_attach, 'var_orientation': var_orientation, 'var_information': var_info})
# RandomcModel.write()
# PlotUOModel.plot_hist_model_pretty('UORandomcSimpleModel', display_title=False)
# PlotUOModel.plot_var_model_pretty('UORandomcSimpleModel', display_title=False)
#
# UniformSimpleModel = UOUniformSimpleModel(root, group, new=True, n_replica=1000)
# p_attach = 0.6
# # a = np.exp(-0.0283)
# # r = 1.7
# # c = round(1-np.sqrt(1-(1-a)/p_attach), 3)
# c = 1
# c_orientation = 1.
# # var_info = (r*p_attach*c*(2-c)-(1-p_attach)*c_orientation**2*np.pi**2/3.) / (p_attach * c ** 2)
# var_info = 0.5
# UniformSimpleModel.run(
#     {'c': c, 'p_attachment': p_attach, 'c_orientation': c_orientation,
#      'var_information': var_info})
# UniformSimpleModel.write()
# PlotUOModel.plot_hist_model_pretty('UOUniformSimpleModel', display_title=False)
# PlotUOModel.plot_var_model_pretty('UOUniformSimpleModel', display_title=False)
#
# CModel = UOCSimpleModel(root, group, new=True, n_replica=10000)
# p_attach = 0.052
# a = np.exp(-0.0283)
# r = 1.7
# var_orientation = 0.09
# var_info = r*(r*p_attach/((1-p_attach)*var_orientation)-1)
# # CModel.run({'p_attachment': p_attach, 'var_orientation': var_orientation, 'var_information': var_info})
# # CModel.write()
#
# PlotUOModel.plot_hist_model_pretty('UOCSimpleModel', display_title=False)
# PlotUOModel.plot_var_model_pretty('UOCSimpleModel', display_title=False)

# group = 'UO'
# ModelAnalysis = UOModelAnalysis(group)
# ModelAnalysis.compute_model_attachment_intervals('UOSimpleModel')

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

#
# ConfidenceModel = UOConfidenceModel(root, group, new=True, n_replica=100000)
# p_att = 0.053
# r = 1.7
# var_orientation = 0.04
#
# var_info = r*(r*p_att/((1-p_att)*var_orientation)-1)
# ConfidenceModel.run({'p_attachment': p_att, 'var_orientation': var_orientation, 'var_information': var_info})
#
# var_info = 1.74
# ConfidenceModel.run({'p_attachment': p_att, 'var_orientation': var_orientation, 'var_information': var_info})
#
# ConfidenceModel.write()
# PlotUOModel.plot_confidence_model_evol()
#
# PlotUOModel.plot_hist_model_pretty('UOConfidenceModel', display_title=False)
# PlotUOModel.plot_var_model_pretty('UOConfidenceModel', display_title=False)

# PersistenceModel = PersistenceModel(root, group, new=True, duration=120, n_replica=1000)
# PersistenceModel.run({'var_orientation': 0.33})
# PersistenceModel.run({'var_orientation': 0.09})
# PersistenceModel.run({'var_orientation': 0.0474})
# PersistenceModel.run({'var_orientation': 0.04})
# PersistenceModel.run({'var_orientation': 0.035})
# PersistenceModel.run({'var_orientation': 0.03})
# PersistenceModel.write()
#
# PlotUOModel.plot_persistence()

# WrappedPersistenceModel = WrappedPersistenceModel(root, group, new=True, duration=120, n_replica=10000)
# WrappedPersistenceModel.run({'kappa_orientation': 10})
# WrappedPersistenceModel.run({'kappa_orientation': 15})
# WrappedPersistenceModel.run({'kappa_orientation': 20})
# WrappedPersistenceModel.run({'kappa_orientation': 30})
# WrappedPersistenceModel.write()
# PlotUOModel.exp = WrappedPersistenceModel.exp
# PlotUOModel.plot_persistence(name='WrappedPersistenceModel')

# UniformPersistenceModel = UniformPersistenceModel(root, group, new=True, duration=20, n_replica=10000)
# # UniformPersistenceModel.run({'c': .15, 'var0': 1})
# UniformPersistenceModel.run({'c': .9, 'var0': 0.1})
# # UniformPersistenceModel.run({'c': .15, 'var0': 0.1})
# # UniformPersistenceModel.run({'c': .13, 'var0': 0.5})
# # UniformPersistenceModel.run({'c': .11, 'var0': 0.5})
# # UniformPersistenceModel.run({'c': .1, 'var0': 0.5})
# # UniformPersistenceModel.run({'c': .09, 'var0': 0.5})
# UniformPersistenceModel.write()
#
# # PlotUOModel.plot_persistence(name='UniformPersistenceModel')
# start_frame_intervals = np.array(np.arange(0, 20, 5), dtype=int)
# end_frame_intervals = start_frame_intervals + 1
# PlotUOModel.plot_hist_model_pretty(
#     'UniformPersistenceModel',
#     start_frame_intervals=start_frame_intervals, end_frame_intervals=end_frame_intervals, fps=1)
# PlotUOModel.plot_var_model_pretty('UniformPersistenceModel')

# PlotUOModel.compare_norm_vonmises()
