import numpy as np
import scipy.stats as scs

from Scripts.root import root
from AnalyseClasses.Models.UOWrappedModels import UOWrappedOutInModelC1
from AnalyseClasses.Models.PlotUOModel import PlotUOModel
from Tools.Plotter.Plotter import Plotter

group = 'UO'
PlotUOModel = PlotUOModel(group)
name_model = 'UOWrappedOutInModelC1'


class Fcts:
    def __init__(self):
        self.plot = PlotUOModel

    @staticmethod
    def run(suff, para_list, n_replica=1000, duration=120, fps=None):
        if fps is None:
            fps = 1
        model = UOWrappedOutInModelC1(
            root, group, new=True, n_replica=n_replica, duration=duration, suff=suff, fps=fps)

        for mu1, mu2, kappa1, kappa2, p1, p2, kappa in para_list:
            model.run(
                {'mu1': mu1, 'mu2': mu2, 'kappa1': kappa1, 'kappa2': kappa2,
                 'p1': p1, 'p2': p2, 'kappa_orientation': kappa})

        model.write()
        PlotUOModel.exp = model.exp

    @staticmethod
    def draw_hist_fit(suff, para_list, adjust, title_option):
        for mu1, mu2, kappa1, kappa2, p1, p2, kappa in para_list:
            para = "(%s, %s, %s, %s, %s, %s, %s)" \
                    % (str(mu1), str(mu2), str(kappa1), str(kappa2), str(p1), str(p2), str(kappa))
            PlotUOModel.plot_hist_fit_model_pretty(
                name_model, para, suff=suff, adjust=adjust, title_option=title_option)

    @staticmethod
    def plot_path_efficiency(suff, para_list, label_option):
        for mu1, mu2, kappa1, kappa2, p1, p2, kappa in para_list:
            para = "(%s, %s, %s, %s, %s, %s, %s)" \
                    % (str(mu1), str(mu2), str(kappa1), str(kappa2), str(p1), str(p2), str(kappa))
            # PlotUOModel.compute_plot_path_efficiency(name_model, para, suff, True, label_option)
            PlotUOModel.compute_plot_path_efficiency_fit(name_model, para, suff, label_option)

    @staticmethod
    def plot_pretty_hist_and_var(suff, adjust, title_option, ):
        dx = 0.2
        start_frame_intervals = np.array(np.arange(0, 2., dx) * 60 * 100, dtype=int)
        end_frame_intervals = np.array(start_frame_intervals + 1000, dtype=int)

        PlotUOModel.plot_hist_model_pretty(
            name_model, title_option=title_option, adjust=adjust, suff=suff,
            start_frame_intervals=start_frame_intervals, end_frame_intervals=end_frame_intervals)

        PlotUOModel.plot_var_model_pretty(name_model, title_option=title_option, adjust=adjust, suff=suff)

    @staticmethod
    def mixture_fct(x, mu1, mu2, kappa1, kappa2, p1, p2, kappa):
        q = p1/(p1+p2)
        p = q
        # p = min(p1, p2)/(p1+p2)
        # print(p, p0)
        if kappa1 == 0:
            if kappa2 == 0:
                y = 1/2/np.pi
            else:
                k = 1/(1/kappa2+1/kappa/p/2)
                y = q/(2*np.pi)+(1-q)*scs.vonmises.pdf(x, loc=mu2, kappa=k)
        else:
            if kappa2 == 0:
                k = 1/(1/kappa1+1/kappa/p/2)
                y = q*scs.vonmises.pdf(x, loc=mu1, kappa=k)+(1-q)/(2*np.pi)
            else:
                k1 = 1/(1/kappa1+1/kappa/p/2)
                k2 = 1/(1/kappa2+1/kappa/p/2)
                y = q*scs.vonmises.pdf(x, loc=mu1, kappa=k1)+(1-q)*scs.vonmises.pdf(x, loc=mu2, kappa=k2)
        return y

    def plot_pretty_hist_and_var_model_c1(self, suff, adjust, title_option):
        dx = 0.2
        start_frame_intervals = np.array(np.arange(0, 2., dx) * 60 * 100, dtype=int)
        end_frame_intervals = np.array(start_frame_intervals + 1000, dtype=int)

        plotter, fig, ax, m = self.plot.plot_hist_model_pretty(
            name_model, title_option=title_option, adjust=adjust, suff=suff,
            start_frame_intervals=start_frame_intervals, end_frame_intervals=end_frame_intervals, save=False)

        if suff is not None:
            name = name_model+'_'+suff
        else:
            name = name_model
        column_names = self.plot.exp.get_data_object(name).get_column_names()
        dx = 0.1
        x = np.arange(0, np.pi+dx, dx)
        for k, column_name in enumerate(column_names):
            i = int(np.floor(k / m))
            j = k % m
            if isinstance(ax, np.ndarray):
                if len(ax.shape) == 1:
                    ax0 = ax[k]
                else:
                    ax0 = ax[i, j]
            else:
                ax0 = ax

            list_para = list(column_name.split(','))
            list_para[0] = list_para[0][1:]
            list_para[-1] = list_para[-1][:-1]
            if len(list_para[1]) == 0:
                list_para.pop()

            mu1, mu2, kappa1, kappa2, p1, p2, kappa = np.array(list_para, dtype=float)

            y = self.mixture_fct(x, mu1, mu2, kappa1, kappa2, p1, p2, kappa)
            ax0.plot(x, y, c='grey', label='guess')

        fig_name = name + '_hist_pretty'
        plotter.save(fig, name=fig_name)
        self.plot.exp.remove_object(name)

        self.plot.plot_var_model_pretty(name_model, title_option=title_option, adjust=adjust, suff=suff)

    @staticmethod
    def get_kappa_bar(kappa_info, p, kappa):
        if kappa_info == 0 or p == 0 or kappa == 0:
            return 0
        else:
            kappa_bar = 1 / (1 / kappa_info + 1 / p / kappa)
            if kappa_bar > 0:
                return kappa_bar
            else:
                raise ValueError('kappa bar not positive')

    @staticmethod
    def get_kappa_info(kappa_bar, p, kappa):
        if kappa_bar == 0 or p == 0 or kappa == 0:
            return 0
        else:
            kappa_info = 1 / (1 / kappa_bar - 1 / p / kappa)
            if kappa_info > 0:
                return kappa_info
            else:
                raise ValueError('kappa info not positive')

    @staticmethod
    def get_p(kappa_bar, kappa_info, kappa):
        if kappa_bar == 0 or kappa_info == 0:
            return 0
        else:
            p = 1 / (kappa * (1 / kappa_bar - 1 / kappa_info))
            if 0 < p < 1:
                return p
            else:
                raise ValueError('probability not between 0 and 1')


def get_results(n_replica):
    kappa = 3
    mu1 = 0
    mu2 = round(np.pi, 2)
    p = 0.2

    kappa_bar_list = [0.001, 0.4, 0.5]
    lg = len(kappa_bar_list)
    suff = 'test'
    para_list = [
        (mu1, mu2, round(Fcts.get_kappa_info(kappa_bar_list[i], p, kappa), 3),
         round(Fcts.get_kappa_info(kappa_bar_list[j], p, kappa), 3), p, p, kappa)
        for i in range(lg) for j in range(lg)]
    # kappa = 3
    # kappa1 = 6
    # p1 = 0.2
    # p2 = 0.06
    # kappa2 = 0.001
    # mu1 = 0
    # mu2 = round(np.pi, 2)
    #
    # Fcts.run(suff, para_list, n_replica=n_replica, fps=100, duration=10)

    adjust = {'top': 0.95, 'right': 0.98, 'left': 0.06, 'bottom': 0.05, 'hspace': 0.3}
    title_option = (r'$(\kappa_1, \kappa_2, p_1, p_2, \kappa)$', [2, 3, 4, 5, 6])
    # title_option = None
    Fcts().plot_pretty_hist_and_var_model_c1(suff, adjust, title_option)


get_results(1000)
