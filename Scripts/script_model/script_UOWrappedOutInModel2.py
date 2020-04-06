import numpy as np

from Scripts.root import root
from AnalyseClasses.Models.UOWrappedModels import UOWrappedOutInModel
from AnalyseClasses.Models.PlotUOModel import PlotUOModel

group = 'UO'
PlotUOModel = PlotUOModel(group)

name_model = 'UOWrappedOutInModel'
WrappedOutInModel = UOWrappedOutInModel(root, group, new=True, n_replica=1000, duration=120)

p_out = 0.07
p_in = p_out*0.33
kappa_orientation = 15
c_outside = 1
c_inside = 1
kappa_info = 2.38

for kappa_info in [2., 3., 5.]:
    for p_in in [p_out, p_out*2/3., p_out*0.33]:
        p_in = round(p_in, 4)
        WrappedOutInModel.run(
            {'c_outside': c_outside, 'c_inside': c_inside, 'p_outside': p_out, 'p_inside': p_in,
             'kappa_orientation': kappa_orientation, 'kappa_information': kappa_info})
WrappedOutInModel.write()


PlotUOModel.exp = WrappedOutInModel.exp
adjust = {'top': 0.92, 'bottom': 0.02, 'left': 0.02, 'right': 0.98}
# title_option = (r'($c_{out}$, $c_{in}$)', [0, 1])
# title_option = (r'$\kappa_{info}$', -1)
# title_option = (r'$p_{inside}$', 3)
title_option = (r'($p_{inside}$, $\kappa_{info}$)', [3, -1])
dx = 0.2
start_frame_intervals = np.array(np.arange(0, 2., dx)*60*100, dtype=int)
end_frame_intervals = np.array(start_frame_intervals + 1000, dtype=int)
PlotUOModel.plot_hist_model_pretty(
    name_model, title_option=title_option, adjust=adjust,
    start_frame_intervals=start_frame_intervals, end_frame_intervals=end_frame_intervals)
PlotUOModel.plot_var_model_pretty(name_model, title_option=title_option, adjust=adjust)
