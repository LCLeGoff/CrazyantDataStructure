import numpy as np

from Scripts.root import root
from AnalyseClasses.Models.UOWrappedModels import UOWrappedSimpleModel
from AnalyseClasses.Models.PlotUOModel import PlotUOModel

group = 'UO'
PlotUOModel = PlotUOModel(group)

WrappedSimpleModel = UOWrappedSimpleModel(root, group, new=True, n_replica=10000)

# p_attach = 0.051
# kappa_orientation = 15
# c = round(1-2*np.pi*0.15/2., 2)
#
# for kappa_info in [0.1, 1, 2, 10, 100]:
#     for c in [0.1, 0.25, 0.5, 0.75, 0.9]:
#         WrappedSimpleModel.run(
#             {'c': c, 'p_attachment': p_attach, 'kappa_orientation': kappa_orientation, 'kappa_information': kappa_info})
#
# WrappedSimpleModel.write()
# PlotUOModel.exp = WrappedSimpleModel.exp

adjust = {'top': 0.97, 'bottom': 0.03, 'left': 0.03, 'right': 0.98, 'hspace': 0.3}
title_option = (r'($c$, $\kappa_{info}$)', [0, 3])
PlotUOModel.plot_hist_model_pretty('UOWrappedSimpleModel', title_option=title_option, adjust=adjust)
PlotUOModel.plot_var_model_pretty('UOWrappedSimpleModel', title_option=title_option, adjust=adjust)

