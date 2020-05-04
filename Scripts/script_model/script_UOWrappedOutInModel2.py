import numpy as np

from Scripts.root import root
from AnalyseClasses.Models.UOWrappedModels import UOWrappedOutInModel2
from AnalyseClasses.Models.PlotUOModel import PlotUOModel

group = 'UO'
PlotUOModel = PlotUOModel(group)

name_model = 'UOWrappedOutInModel2'
# suff = 'c1_2'
suff = 'test'
# suff = 'c1'
WrappedOutInModel = UOWrappedOutInModel2(root, group, new=True, n_replica=1000, suff=suff)

kappa_orientation = 15
c = 1
for kappa_info in [4, 5, 6]:
    WrappedOutInModel.run({'c_outside': c, 'c_inside': c,
                           'kappa_orientation': kappa_orientation, 'kappa_information': kappa_info})
WrappedOutInModel.write()
PlotUOModel.exp = WrappedOutInModel.exp

# adjust = {'top': 0.92, 'right': 0.99}
adjust = {'top': 0.98, 'right': 0.99, 'left': 0.15, 'bottom': 0.14}
title_option = None
# title_option = (r'$\kappa_{info}$', -1)
# title_option = (r'$c_{in}$', 1)
# title_option = (r'$(c, \kappa_{info})$', [0, -1])
dx = 0.2
start_frame_intervals = np.array(np.arange(0, 2., dx)*60*100, dtype=int)
end_frame_intervals = np.array(start_frame_intervals + 1000, dtype=int)
PlotUOModel.plot_hist_model_pretty(
    name_model, title_option=title_option, adjust=adjust, suff=suff,
    start_frame_intervals=start_frame_intervals, end_frame_intervals=end_frame_intervals)
PlotUOModel.plot_var_model_pretty(name_model, title_option=title_option, adjust=adjust, suff=suff)
