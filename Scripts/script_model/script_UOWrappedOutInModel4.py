import numpy as np

from Scripts.root import root
from AnalyseClasses.Models.UOWrappedModels import UOWrappedOutInModel4
from AnalyseClasses.Models.PlotUOModel import PlotUOModel

group = 'UO'
PlotUOModel = PlotUOModel(group)
name_model = 'UOWrappedOutInModel4'


class Fcts:
    def __init__(self):
        pass

    @staticmethod
    def run(suff, para_list, n_replica=1000, duration=120, fps=None, category=None):
        if fps is None:
            fps = 1
        if category is None:
            category = 'Models'
        model = UOWrappedOutInModel4(
            root, group, new=True, n_replica=n_replica, duration=duration, suff=suff, fps=fps, category=category)

        for kappa_orient, c, kappa_info, u_out, q in para_list:
            model.run(
                {'c_outside': c, 'c_inside': c, 'kappa_orientation': kappa_orient,
                 'kappa_information': kappa_info, 'u_out': u_out, 'q': q})

        model.write()
        PlotUOModel.exp = model.exp

    @staticmethod
    def plot_pretty_hist_and_var(suff, adjust, title_option):
        dx = 0.2
        start_frame_intervals = np.array(np.arange(0, 2., dx) * 60 * 100, dtype=int)
        end_frame_intervals = np.array(start_frame_intervals + 1000, dtype=int)

        PlotUOModel.plot_hist_model_pretty(
            name_model, title_option=title_option, adjust=adjust, suff=suff,
            start_frame_intervals=start_frame_intervals, end_frame_intervals=end_frame_intervals)

        PlotUOModel.plot_var_model_pretty(name_model, title_option=title_option, adjust=adjust, suff=suff)


def show_results(n_replica, fps=None):
    q = 0.58

    kappa_orientation_list = [15]
    kappa_info_list = [2.11, 2.53]
    u_out_list = [0.3, 0.4, 0.5]
    c_list = [1.0]

    para_list = [
        (kappa_orientation, c, kappa_info, u_out, q)
        for kappa_orientation in kappa_orientation_list
        for kappa_info in kappa_info_list
        for c in c_list
        for u_out in u_out_list
    ]
    suff = '_test'
    if fps is not None:
        suff += '_'+str(int(fps))+'fps'

    Fcts.run(suff, para_list, n_replica=n_replica, fps=fps, category='Models2')

    title_option = None
    # title_option = (['c', 'kappa', 'info', 'u', 'q'], [1, 2, 3, 4, 5])

    adjust = {'top': 0.95, 'right': 0.98, 'left': 0.06, 'bottom': 0.05, 'hspace': 0.3}
    Fcts.plot_pretty_hist_and_var(suff, adjust, title_option)


show_results(1000, fps=100)
