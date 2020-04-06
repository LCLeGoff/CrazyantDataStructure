import numpy as np

from Scripts.root import root
from AnalyseClasses.Models.UOModels import UOSimpleModel
from AnalyseClasses.Models.PlotUOModel import PlotUOModel

group = 'UO'
PlotUOModel = PlotUOModel(group)


SimpleModel = UOSimpleModel(root, group, new=True, n_replica=10000)

p_attach = 0.057
a = np.exp(-0.0283)
r = 1.7
var_orientation = 0.04

c = round(1-np.sqrt(1-(1-a)/p_attach), 3)
var_info = (r*p_attach*c*(2-c)-(1-p_attach)*var_orientation)/(p_attach*c**2)
SimpleModel.run({'c': c, 'p_attachment': p_attach, 'var_orientation': var_orientation, 'var_information': var_info})

c = 0.43
SimpleModel.run({'c': c, 'p_attachment': p_attach, 'var_orientation': var_orientation, 'var_information': var_info})
# #
# # var_orientation = 0.09
# # var_info = r*(r*p_attach/((1-p_attach)*var_orientation)-1)
# # c = r/(r+var_info)
# # SimpleModel.run({'c': c, 'p_attachment': p_attach, 'var_orientation': var_orientation, 'var_information': var_info})
# #
# # var_orientation = 0.09
# # c = 0.9
# # var_info = (r*p_attach*c*(2-c)-(1-p_attach)*var_orientation)/(p_attach*c**2)
# # SimpleModel.run({'c': c, 'p_attachment': p_attach, 'var_orientation': var_orientation, 'var_information': var_info})
# #
SimpleModel.write()
PlotUOModel.exp = SimpleModel.exp

PlotUOModel.plot_hist_model_pretty('UOSimpleModel', title_option=('c', 0), adjust={'top': 0.92, 'right': 0.99})
PlotUOModel.plot_var_model_pretty(
    'UOSimpleModel', title_option=('c', 0), adjust={'top': 0.92, 'right': 0.99}, plot_fisher=True)


# PlotUOModel.plot_simple_model_evol()
# PlotUOModel.plot_indiv_hist_evol('UOSimpleModel', 0)
