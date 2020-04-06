import numpy as np

from AnalyseClasses.Models.UOModelAnalysis import UOModelAnalysis
from Scripts.root import root
from AnalyseClasses.Models.UOModels import UOSimpleModel, UOConfidenceModel, UORWModel, UOOutsideModel, UORandomcSimpleModel, UOCSimpleModel, \
    UOUniformSimpleModel
from AnalyseClasses.Models.UOWrappedModels import UOWrappedSimpleModel
from AnalyseClasses.Models.UOPersistenceModels import PersistenceModel, WrappedPersistenceModel, UniformPersistenceModel
from AnalyseClasses.Models.PlotUOModel import PlotUOModel

group = 'UO'
PlotUOModel = PlotUOModel(group)

ConfidenceModel = UOConfidenceModel(root, group, new=True, n_replica=10000)
# p_att = 0.057
# r = 1.7
# var_orientation = 0.04
#
# var_info = round(r*(r*p_att/((1-p_att)*var_orientation)-1), 2)
# ConfidenceModel.run({'p_attachment': p_att, 'var_orientation': var_orientation, 'var_information': var_info})
#
# var_info = 2.1
# ConfidenceModel.run({'p_attachment': p_att, 'var_orientation': var_orientation, 'var_information': var_info})
#
# ConfidenceModel.write()
# PlotUOModel.plot_confidence_model_evol()

PlotUOModel.plot_hist_model_pretty(
    'UOConfidenceModel', title_option=(r'$\sigma^2_{info}$', -1), adjust={'top': 0.915, 'right': 0.99})
PlotUOModel.plot_var_model_pretty(
    'UOConfidenceModel', title_option=(r'$\sigma^2_{info}$', -1), adjust={'top': 0.915, 'right': 0.99}, plot_fisher=True)
