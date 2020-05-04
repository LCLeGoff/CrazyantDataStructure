import numpy as np

from Scripts.root import root
from AnalyseClasses.Models.UOWrappedModels import UOWrappedOutInModel3
from AnalyseClasses.Models.PlotUOModel import PlotUOModel

group = 'UO'
PlotUOModel = PlotUOModel(group)
name_model = 'UOWrappedOutInModel3'


class Fcts:
    def __init__(self):
        pass

    @staticmethod
    def run(suff, para_list, n_replica=1000, duration=120):
        model = UOWrappedOutInModel3(
            root, group, new=True, n_replica=n_replica, duration=duration, suff=suff, fps=1)

        for kappa_orient, c, k_i, p_l in para_list:
            model.run(
                {'c_outside': c, 'c_inside': c, 'kappa_orientation': kappa_orient,
                 'kappa_information': k_i, 'prop_leading': p_l})

        model.write()
        PlotUOModel.exp = model.exp

    @staticmethod
    def draw_hist_fit(suff2, para_list, adjust, title_option):
        for kappa_orient, c, k_i, p_l in para_list:
            para2 = "(%s, %s, %s, %s, %s)" % (str(c), str(c), str(kappa_orient), str(k_i), str(p_l))
            PlotUOModel.plot_hist_fit_model_pretty(
                name_model, para2, suff=suff2, adjust=adjust, title_option=title_option)

    @staticmethod
    def plot_path_efficiency(suff, para_list, label_option):
        for kappa_orient, c, k_i, p_l in para_list:
            para = "(%s, %s, %s, %s, %s)" % (str(c), str(c), str(kappa_orient), str(k_i), str(p_l))
            # PlotUOModel.compute_plot_path_efficiency(name_model, para, suff, True, label_option)
            PlotUOModel.compute_plot_path_efficiency_fit(name_model, para, suff, label_option)

    @staticmethod
    def plot_pretty_hist_and_var(suff, adjust, title_option):
        dx = 0.2
        start_frame_intervals = np.array(np.arange(0, 2., dx) * 60 * 100, dtype=int)
        end_frame_intervals = np.array(start_frame_intervals + 1000, dtype=int)

        PlotUOModel.plot_hist_model_pretty(
            name_model, title_option=title_option, adjust=adjust, suff=suff,
            start_frame_intervals=start_frame_intervals, end_frame_intervals=end_frame_intervals)

        PlotUOModel.plot_var_model_pretty(name_model, title_option=title_option, adjust=adjust, suff=suff)


def show_previous_results():
    kappa_orientation = 15
    c_list = [0.5, 0.9, 0.95, 1.]

    for c in c_list:
        suff = 'c'+str(c)+'_kappa_orient'+str(kappa_orientation)

        prop_list = [0.05, 0.1, 0.2]
        kappa_info_list = [3, 4, 5]
        para_list = [(kappa_orientation, c, k_i, p_l) for k_i in kappa_info_list for p_l in prop_list]

        Fcts.run(suff, para_list, n_replica=5000)

        adjust = {'top': 0.95, 'right': 0.98, 'left': 0.06, 'bottom': 0.05, 'hspace': 0.3}
        title_option = (r'$(c, \kappa_{info}, u)$', [0, -2, -1])
        Fcts.plot_pretty_hist_and_var(suff, adjust, title_option)


def show_previous_results2():
    kappa_orientation = 15
    c_list = [0.5, 0.9, 0.95, 1.]

    suff = 'kappa_orient'+str(kappa_orientation)
    para_list = [(kappa_orientation, c, 4, 0.1) for c in c_list]

    Fcts.run(suff, para_list, n_replica=5000)
    #
    # title_option = ('c', 0)
    #
    # adjust = {'top': 0.95, 'right': 0.98, 'left': 0.06, 'bottom': 0.05, 'hspace': 0.3}
    # Fcts.plot_pretty_hist_and_var(suff, adjust, title_option)
    #
    # adjust = {'top': 0.92, 'right': 0.99, 'hspace': 0.3, 'left': 0.05, 'bottom': 0.07}
    # Fcts.draw_hist_fit(suff, para_list, adjust, title_option=title_option)
    #
    # PlotUOModel.compute_path_efficiency(name_model, suff=suff, redo=True, label_option=title_option)

    for kappa_orient, c, k_i, p_l in para_list:
        para = "(%s, %s, %s, %s, %s)" % (str(c), str(c), str(kappa_orient), str(k_i), str(p_l))
        PlotUOModel.compute_plot_attachment_prop_fit(name_model, para, suff)


def show_path_efficiency():
    kappa_orient_list = [1, 2, 3, 4, 15]

    suff = 'kappa_orients'
    para_list = [(kappa_orient, 1., 4, 0.1) for kappa_orient in kappa_orient_list]

    Fcts.run(suff, para_list, n_replica=1000)

    title_option = (r'$\kappa_{orient}$', 2)
    PlotUOModel.compute_path_efficiency(name_model, suff=suff, redo=True, label_option=title_option)


def show_new_results():
    kappa_orientation = 2
    c_list = [0.5, 0.9, 0.95, 1.]

    for c in c_list:
        suff = 'c'+str(c)+'_kappa_orient'+str(kappa_orientation)

        prop_list = [0.2, 0.3, 0.4, 0.5]
        kappa_info_list = [2, 3, 4, 5, 6, 7]
        para_list = [(kappa_orientation, c, k_i, p_l) for p_l in prop_list for k_i in kappa_info_list]

        Fcts.run(suff, para_list, n_replica=2500)

        adjust = {'top': 0.95, 'right': 0.98, 'left': 0.06, 'bottom': 0.05, 'hspace': 0.3}
        title_option = (r'$(c, \kappa_{info}, u)$', [0, -2, -1])
        Fcts.plot_pretty_hist_and_var(suff, adjust, title_option)


def show_new_results2():
    kappa_orientation = 2
    c_list = [0.5, 0.9, 0.95, 1.]

    para_list = [(kappa_orientation, c, 4, 0.5) for c in c_list]
    suff = 'kappa_orient'+str(kappa_orientation)

    Fcts.run(suff, para_list, n_replica=5000)

    title_option = ('c', 0)

    adjust = {'top': 0.95, 'right': 0.98, 'left': 0.06, 'bottom': 0.05, 'hspace': 0.3}
    Fcts.plot_pretty_hist_and_var(suff, adjust, title_option)

    adjust = {'top': 0.92, 'right': 0.99, 'hspace': 0.3, 'left': 0.05, 'bottom': 0.07}
    Fcts.draw_hist_fit(suff, para_list, adjust, title_option=title_option)

    PlotUOModel.compute_path_efficiency(name_model, suff=suff, redo=True, label_option=title_option)
    Fcts.plot_path_efficiency(suff, para_list, label_option=title_option)

    for kappa_orient, c, k_i, p_l in para_list:
        para = "(%s, %s, %s, %s, %s)" % (str(c), str(c), str(kappa_orient), str(k_i), str(p_l))
        PlotUOModel.compute_plot_attachment_prop_fit(name_model, para, suff)


# show_previous_results()
# show_previous_results2()
# show_path_efficiency()
# show_new_results()
# show_new_results2()

