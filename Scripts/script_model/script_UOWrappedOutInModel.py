import numpy as np

from Scripts.root import root
from AnalyseClasses.Models.UOWrappedModels import UOWrappedOutInModel
from AnalyseClasses.Models.PlotUOModel import PlotUOModel

group = 'UO'
PlotUOModel = PlotUOModel(group)

name_model = 'UOWrappedOutInModel'

# for c in [0.25, 0.5, 0.75]:
for c in [1]:
    suff = 'c'+str(c)
    WrappedOutInModel = UOWrappedOutInModel(root, group, new=True, n_replica=5000, duration=150, suff=suff)

    p_out = 0.057
    kappa_orientation = 15

    # for p_in in [0.02, p_out, 0.07]:
    for p_in in [0.73*p_out]:
        for kappa_info in [2, 3, 5, 8]:
            p_in = round(p_in, 4)
            WrappedOutInModel.run(
                {'c_outside': c, 'c_inside': c, 'p_outside': p_out, 'p_inside': p_in,
                 'kappa_orientation': kappa_orientation, 'kappa_information': kappa_info})
    WrappedOutInModel.write()
    PlotUOModel.exp = WrappedOutInModel.exp

    # adjust = {'top': 0.97, 'bottom': 0.05, 'left': 0.05, 'right': 0.98, 'hspace': 0.3}
    adjust = {'top': 0.92, 'bottom': 0.09, 'left': 0.08, 'right': 0.95, 'hspace': 0.3}
    title_option = (r'$\kappa_{info}$', -1)
    # title_option = (r'($p_{inside}$, $\kappa_{info}$)', [3, -1])
    dx = 0.2
    start_frame_intervals = np.array(np.arange(0, 2., dx)*60*100, dtype=int)
    end_frame_intervals = np.array(start_frame_intervals + 1000, dtype=int)
    PlotUOModel.plot_hist_model_pretty(
        name_model, suff=suff, title_option=title_option, adjust=adjust,
        start_frame_intervals=start_frame_intervals, end_frame_intervals=end_frame_intervals)
    PlotUOModel.plot_var_model_pretty(name_model, suff=suff, title_option=title_option, adjust=adjust)
