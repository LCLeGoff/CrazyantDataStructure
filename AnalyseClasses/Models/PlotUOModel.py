import numpy as np
import pandas as pd
from scipy import stats as scs

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from AnalyseClasses.Models.UOWrappedModels import UOWrappedSimpleModel
from DataStructure.VariableNames import id_exp_name
from Scripts.root import root
from Tools.MiscellaneousTools import Geometry as Geo, Fits
from Tools.MiscellaneousTools.ArrayManipulation import get_index_interval_containing
from Tools.Plotter.BasePlotters import BasePlotters
from Tools.Plotter.Plotter import Plotter

import matplotlib as mlt
mlt.rcParams['axes.titlesize'] = 19
mlt.rcParams['axes.labelsize'] = 15


class PlotUOModel(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'Models'
        self.model = None

    def get_model(self, name):
        if name == 'UOWrappedSimpleModel':
            self.model = UOWrappedSimpleModel(root, 'UO', new=False)

    def plot_simple_model_evol(self, suff=None, n=None, m=None):

        name = 'UOSimpleModel'
        self._plot_hist_evol(name, n, m, 'simple', suff)
        self._plot_var_evol(name, n, m, 'simple', suff)

    def plot_outside_model_evol(self, suff=None, n=None, m=None):

        name = 'UOOutsideModel'
        self._plot_hist_evol(name, n, m, 'outside', suff)
        self._plot_var_evol(name, n, m, 'outside', suff)

    def plot_rw_model_evol(self, suff=None, n=None, m=None):

        name = 'UORWModel'
        self._plot_hist_evol(name, n, m, 'simple', suff)
        self._plot_var_evol(name, n, m, 'simple', suff)

    def plot_confidence_model_evol(self, suff=None, n=None, m=None):

        name = 'UOConfidenceModel'
        self._plot_hist_evol(name, n, m, 'confidence', suff)
        self._plot_var_evol(name, n, m, 'confidence', suff)
        self._plot_confidence_evol(name+'_confidence', n, m, 'confidence', suff)

    def plot_persistence(self, name='PersistenceModel', suff=None):

        self.__plot_cosinus_correlation_vs_length(name, suff)
        # self.__plot_cosinus_correlation_vs_arclength(name, suff)

    def __plot_cosinus_correlation_vs_length(self, name, suff=None):

        self.exp.load(name, reload=False)
        column_names = self.exp.get_data_object(name).get_column_names()

        n_replica = max(self.exp.get_df(name).index.get_level_values(id_exp_name))
        duration = max(self.exp.get_df(name).index.get_level_values('t'))

        speed = 1
        index_values = np.arange(duration + 1)*speed/10.
        index_values2 = np.arange(duration + 1)

        self.exp.add_new_empty_dataset('plot', index_names='lag', column_names=column_names,
                                       index_values=index_values,
                                       fill_value=0, category=self.category, replace=True)
        self.exp.add_new_empty_dataset('plot2', index_names='lag', column_names=column_names,
                                       index_values=index_values2,
                                       fill_value=0, category=self.category, replace=True)

        for column_name in column_names:
            df = pd.DataFrame(self.exp.get_df(name)[column_name])

            for id_exp in range(1, max(df.index.get_level_values(id_exp_name))+1):
                df2 = df.loc[id_exp, :]

                orientations = Geo.norm_angle(df2)

                res = np.zeros(len(orientations))
                weight = np.zeros(len(orientations))

                for i in range(1, len(orientations)):
                    res[:-i] += np.cos(Geo.angle_distance(orientations[i], orientations[i:])).ravel()
                    weight[:-i] += 1.

                res /= weight

                self.exp.get_df('plot')[column_name] += res
                self.exp.get_df('plot2')[column_name] += res

        self.exp.get_data_object('plot').df /= float(n_replica)
        self.exp.get_data_object('plot2').df /= float(n_replica)


        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('plot'))
        fig, ax = plotter.plot(xlabel='Distance along trajectory (cm)', ylabel='Cosine correlation', marker=None)
        ax.axhline(0, ls='--', c='grey')
        ax.grid()

        if suff is None:
            plotter.save(fig, name=name+'_length')
        else:
            plotter.save(fig, name=name+'_length_'+suff)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('plot2'))
        fig, ax = plotter.plot(xlabel='', ylabel='Cosine correlation', marker=None)
        ax.axhline(0, ls='--', c='grey')
        ax.axhline(0.5, ls='--', c='grey')
        ax.grid()

        if suff is None:
            plotter.save(fig, name=name+'_length2')
        else:
            plotter.save(fig, name=name+'_length2_'+suff)

    def __plot_cosinus_correlation_vs_arclength(self, name, suff=None):

        self.exp.load(name)
        column_names = self.exp.get_data_object(name).get_column_names()

        radius = 0.85

        dtheta = 1.
        index_values = np.arange(0, 500, dtheta)

        self.exp.add_new_empty_dataset('plot', index_names='lag', column_names=column_names,
                                       index_values=index_values,
                                       fill_value=0, category=self.category, replace=True)

        for column_name in column_names:
            df = pd.DataFrame(self.exp.get_df(name)[column_name])
            norm = np.zeros(len(index_values))

            for id_exp in range(1, max(df.index.get_level_values(id_exp_name))+1):
                df2 = df.loc[id_exp, :]
                orientations = np.array(df2).ravel()
                d_orientations = Geo.angle_distance(orientations[1:], orientations[:-1])
                arclength = np.cumsum(np.abs(d_orientations)) * radius

                orientations2 = np.zeros(len(index_values))
                idx = 0
                for i, arc in enumerate(arclength):
                    idx = get_index_interval_containing(arc, index_values)
                    orientations2[idx:] = orientations[i]

                orientations2 = orientations2[:idx+1]

                corr = np.zeros(len(orientations2))
                weight = np.zeros(len(orientations2))

                for i in range(1, len(orientations2)):
                    corr[:-i] += np.cos(orientations2[i] - orientations2[i:]).ravel()
                    weight[:-i] += 1.

                corr2 = np.zeros(len(index_values))
                corr2[:len(corr)] = corr / weight

                norm[:len(orientations2)] += 1
                self.exp.get_df('plot')[column_name] += corr2

            self.exp.get_df('plot')[column_name] /= norm

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('plot'))
        fig, ax = plotter.plot(xlabel='Arclength along trajectory (cm)', ylabel='Cosine correlation', marker=None)
        ax.axhline(0, ls='--', c='grey')
        ax.set_xlim((0, 12.5))
        ax.set_ylim((-0.1, 1.1))
        ax.grid()

        if suff is None:
            plotter.save(fig, name=name+'_arclength')
        else:
            plotter.save(fig, name=name+'_arclength_'+suff)

    def _plot_confidence_evol(self, name, n=None, m=None, model=None, suff=None):

        self.exp.load(name)

        column_names = self.exp.get_data_object(name).get_column_names()

        if n is None:
            n = int(np.floor(np.sqrt(len(column_names))))
            m = int(np.ceil(len(column_names) / n))

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(name))
        fig, ax = plotter.create_plot(figsize=(4 * m, 4.2 * n), nrows=n, ncols=m, top=0.9, bottom=0.05)

        for k, column_name in enumerate(column_names):
            i = int(k / m)
            j = k % m

            if isinstance(ax, np.ndarray):
                if len(ax.shape) == 1:
                    ax0 = ax[k]
                else:
                    ax0 = ax[i, j]
            else:
                ax0 = ax

            list_id_exp = set(self.exp.get_df(name).index.get_level_values(id_exp_name))
            df = self.exp.get_df(name).loc[1, :]
            df2 = df[column_name]
            for id_exp in range(2, max(list_id_exp)):
                df = self.exp.get_df(name).loc[id_exp, :]
                df2 += df[column_name]

            self.exp.add_new_dataset_from_df(df=df2/float(len(list_id_exp)), name='temp',
                                             category=self.category, replace=True)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
            plotter.plot(xlabel='orientation', ylabel='PDF', preplot=(fig, ax0),
                         title=column_name, label='confidence mean', marker=None)

            ax0.legend()

        if model == 'simple':
            fig.suptitle(r"Simple model, parameters $(c, p_{att}, \sigma_{orient}, \sigma_{info})$",
                         fontsize=15)
        elif model == 'confidence':
            fig.suptitle(r"Confidence model, parameters $(p_{att}, \sigma_{orient}, \sigma_{info})$",
                         fontsize=15)

        if suff is None:
            plotter.save(fig, name=name)
        else:
            plotter.save(fig, name=name+'_'+suff)

    def _plot_hist_evol(self, name, n=None, m=None, model=None, suff=None):

        experimental_name = 'food_direction_error_hist_evol'
        experimental_name_attach = 'food_direction_error_hist_evol_around_first_attachment'
        self.exp.load([name, experimental_name, experimental_name_attach])

        self.exp.load(name)
        self.exp.get_data_object(name).df = np.abs(self.exp.get_df(name))

        time_name = 't'
        column_names = self.exp.get_data_object(name).get_column_names()

        dx = 0.25
        start_time_intervals = np.arange(0, 4., dx)*60*100
        end_time_intervals = start_time_intervals + dx*60*100*2

        dtheta = np.pi / 25.
        bins = np.arange(0, np.pi, dtheta)

        if n is None:
            n = int(np.floor(np.sqrt(len(column_names))))
            m = int(np.ceil(len(column_names) / n))

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(name))
        fig, ax = plotter.create_plot(figsize=(4 * m, 4 * n), nrows=n, ncols=m, top=0.85, bottom=0.1, left=0.05)

        plotter_exp = Plotter(root=self.exp.root, obj=self.exp.get_data_object(experimental_name))
        fig2, ax2 = plotter_exp.create_plot(figsize=(4 * m, 4 * n), nrows=n, ncols=m, top=0.85, bottom=0.1, left=0.05)

        plotter_exp2 = Plotter(root=self.exp.root, obj=self.exp.get_data_object(experimental_name_attach))
        fig3, ax3 = plotter.create_plot(figsize=(4 * m, 4 * n), nrows=n, ncols=m, top=0.85, bottom=0.1, left=0.05)

        for k, column_name in enumerate(column_names):

            hist_name = self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_time_intervals,
                                                  end_frame_intervals=end_time_intervals, bins=bins,
                                                  index_name=time_name, column_to_hist=column_name, replace=True)
            # self.exp.get_df(hist_name).index = self.exp.get_index(hist_name)**2

            i = int(np.floor(k / m))
            j = k % m
            if isinstance(ax, np.ndarray):
                if len(ax.shape) == 1:
                    ax0 = ax[k]
                    ax20 = ax2[k]
                    ax30 = ax3[k]
                else:
                    ax0 = ax[i, j]
                    ax20 = ax2[i, j]
                    ax30 = ax3[i, j]
            else:
                ax0 = ax
                ax20 = ax2
                ax30 = ax3

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
            plotter.plot(xlabel='orientation', ylabel='PDF',
                         normed=True, preplot=(fig, ax0), title=column_name)

            plotter_exp.plot(xlabel='orientation', ylabel='PDF', marker='',
                             normed=True, preplot=(fig2, ax20), title=column_name)

            plotter_exp2.plot(xlabel='orientation', ylabel='PDF', marker='',
                              normed=True, preplot=(fig3, ax30), title=column_name)

            if model == 'simple':

                c = float(column_name.split(',')[0][1:])
                p_attach = float(column_name.split(',')[1])
                var_orientation = float(column_name.split(',')[2])
                var_info = float(column_name.split(',')[3][:-1])

                b = var_orientation+p_attach*c**2*var_info
                a = 1-p_attach*c*(2-c)
                r = b/(1-a)

                x = self.exp.get_df(hist_name).index

                ax0.plot(x, 2*scs.norm.pdf(x, scale=np.sqrt(r)))
                ax20.plot(x, 2*scs.norm.pdf(x, scale=np.sqrt(r)))
                ax30.plot(x, 2*scs.norm.pdf(x, scale=np.sqrt(r)))

        if model == 'simple':
            fig.suptitle(r"$(c, p_{att}, \sigma_{orient}, \sigma_{info})$ = ",
                         fontsize=15)
            fig2.suptitle(r"$(c, p_{att}, \sigma_{orient}, \sigma_{info})$ = ",
                          fontsize=15)
            fig3.suptitle(r"$(c, p_{att}, \sigma_{orient}, \sigma_{info})$ = ",
                          fontsize=15)
        if model == 'outside':
            fig.suptitle(r"$(c, \sigma_{orient}, \sigma_{info})$ = ",
                         fontsize=15)
        elif model == 'confidence':
            fig.suptitle(r"Confidence model, parameters $(p_{att}, \sigma_{orient}, \sigma_{info})$",
                         fontsize=15)

        if suff is None:
            fig_name = name + '_hist'
        else:
            fig_name = name + '_hist_' + suff
        plotter.save(fig, name=fig_name)
        plotter.save(fig2, name=fig_name+'_experiment')
        plotter.save(fig3, name=fig_name+'_experiment_around_first_attachment')

        self.exp.remove_object(name)

    def plot_indiv_hist_evol(self, name, column_num, model='simple', suff=None):

        experimental_name = 'food_direction_error_hist_evol'
        experimental_name_attach = 'food_direction_error_hist_evol_around_first_attachment'
        self.exp.load([name, experimental_name, experimental_name_attach])

        self.exp.load(name)
        self.exp.get_data_object(name).df = np.abs(self.exp.get_df(name))

        time_name = 't'
        column_name = self.exp.get_columns(name)[column_num]

        dx = 0.25
        start_time_intervals = np.arange(0, 4., dx)*60*100
        end_time_intervals = start_time_intervals + dx*60*100*2

        dtheta = np.pi / 25.
        bins = np.arange(0, np.pi, dtheta)

        n = m = 4

        plotter_to_save = Plotter(root=self.exp.root, obj=self.exp.get_data_object(name))
        fig, ax = plotter_to_save.create_plot(figsize=(4 * m, 4 * n), nrows=n, ncols=m, top=0.9, bottom=0.05, left=0.05)

        hist_name = self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_time_intervals,
                                              end_frame_intervals=end_time_intervals, bins=bins,
                                              index_name=time_name, column_to_hist=column_name, replace=True)
        column_names = self.exp.get_columns(hist_name)

        for k, column_name2 in enumerate(column_names):

            i = int(np.floor(k / m))
            j = k % m
            ax0 = ax[i, j]

            self.exp.add_new_dataset_from_df(self.exp.get_df(hist_name)[column_name2], 'temp', replace=True)
            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
            plotter.plot(xlabel='orientation', ylabel='PDF', c='k', marker='', ls='--',
                         normed=True, preplot=(fig, ax0), label='Model', title=r't$\in$' + column_name2)

            if column_name2 in self.exp.get_columns(experimental_name):
                self.exp.add_new_dataset_from_df(self.exp.get_df(experimental_name)[column_name2], 'temp', replace=True)
                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
                plotter.plot(xlabel='orientation', ylabel='PDF', marker='', c='red',
                             normed=True, preplot=(fig, ax0), label='Exp', title=r't$\in$' + column_name2)

            if column_name2 in self.exp.get_columns(experimental_name_attach):
                self.exp.add_new_dataset_from_df(
                    self.exp.get_df(experimental_name_attach)[column_name2], 'temp', replace=True)
                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
                plotter.plot(xlabel='orientation', ylabel='PDF', c='orange', marker='',
                             normed=True, preplot=(fig, ax0), label='Aligned Exp', title=r't$\in$' + column_name2)

            if model == 'simple':

                c = float(column_name.split(',')[0][1:])
                p_attach = float(column_name.split(',')[1])
                var_orientation = float(column_name.split(',')[2])
                var_info = float(column_name.split(',')[3][:-1])
                var0 = np.pi**2/3.

                b = var_orientation+p_attach*c**2*var_info
                a = 1-p_attach*c*(2-c)
                r = b/(1-a)

                t0 = float(column_name2.split(',')[0][1:])
                t1 = float(column_name2.split(',')[1][:-1])
                t = (t0+t1)/2.
                var = a**t*(var0-r)+r

                x = self.exp.get_df(hist_name).index

                ax0.plot(x, 2*scs.norm.pdf(x, scale=np.sqrt(var)), label='Theory', c='w', ls='--')
            ax0.legend()

        if suff is None:
            fig_name = name + 'indiv_hist'
        else:
            fig_name = name + 'indiv_hist_' + suff
        plotter_to_save.save(fig, name=fig_name)
        self.exp.remove_object(name)

    def _plot_var_evol(self, name, n=None, m=None, model=None, suff=None):

        experimental_name = 'food_direction_error_var_evol'
        experimental_name_attach = 'food_direction_error_var_evol_around_first_attachment'
        self.exp.load([name, experimental_name, experimental_name_attach])

        time_name = 't'
        column_names = self.exp.get_data_object(name).get_column_names()

        dx = 0.1
        dx2 = 0.01
        start_time_intervals = np.arange(0, 3., dx2)*60*100
        end_time_intervals = start_time_intervals + dx*60*100*2

        if n is None:
            n = int(np.floor(np.sqrt(len(column_names))))
            m = int(np.ceil(len(column_names) / n))

        plotter_experiment = Plotter(
            root=self.exp.root, obj=self.exp.get_data_object(experimental_name), category=self.category)
        plotter_experiment_attach = Plotter(
            root=self.exp.root, obj=self.exp.get_data_object(experimental_name_attach), category=self.category)
        fig, ax = plotter_experiment.create_plot(figsize=(4 * m, 4 * n), nrows=n, ncols=m, top=0.85, bottom=0.01)

        for k, column_name in enumerate(column_names):

            var_name = self.exp.variance_evolution(name_to_var=name,  start_index_intervals=start_time_intervals,
                                                   end_index_intervals=end_time_intervals, index_name=time_name,
                                                   column_to_var=column_name, replace=True)
            self.exp.get_df(var_name).index = self.exp.get_index(var_name)/100.

            i = int(np.floor(k / m))
            j = k % m
            if isinstance(ax, np.ndarray):
                if len(ax.shape) == 1:
                    ax0 = ax[k]
                else:
                    ax0 = ax[i, j]
            else:
                ax0 = ax

            if model == 'simple':
                c = float(column_name.split(',')[0][1:])
                p_attach = float(column_name.split(',')[1])
                var_orientation = float(column_name.split(',')[2])
                var_info = float(column_name.split(',')[3][:-1])

                def variance(t):
                    b = var_orientation+p_attach*c**2*var_info
                    a = 1-p_attach*c*(2-c)
                    r = b/(1-a)
                    s = np.pi**2/3.
                    return a**t*(s-r)+r

                t_tab = np.array(self.exp.get_df(var_name).index)
                ax0.plot(t_tab, variance(t_tab), label='Theory')

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(var_name))
            plotter.plot(xlabel='Time', ylabel='Variance', preplot=(fig, ax0), title=column_name, label='Model')
            plotter_experiment.plot(
                xlabel='Time', ylabel='Variance', preplot=(fig, ax0), c='grey', title=column_name, label='exp')
            plotter_experiment_attach.plot(
                xlabel='Time', ylabel='Variance', preplot=(fig, ax0), c='w', title=column_name, label='exp 2')

            ax0.legend()
            plotter.draw_vertical_line(ax0)

        if model == 'simple':
            fig.suptitle(r"$(c, p_{att}, \sigma_{orient}, \sigma_{info})$ = ", fontsize=15)
        elif model == 'outside':
            fig.suptitle(r"$(c, \sigma_{orient}, \sigma_{info})$ = ", fontsize=15)
        elif model == 'confidence':
            fig.suptitle(r"Confidence model, parameters $(p_{att}, \sigma_{orient}, \sigma_{info})$", fontsize=15)

        if suff is None:
            plotter_experiment.save(fig, name=name+'_var')
        else:
            plotter_experiment.save(fig, name=name+'_var_'+suff)

    def plot_hist_model_pretty(
            self, name, n=None, m=None, title_option=None, suff=None,
            start_frame_intervals=None, end_frame_intervals=None, fps=100., adjust=None):

        if suff is not None:
            name += '_'+suff
        steady_state_name = 'food_direction_error_hist_steady_state'
        self.exp.load([name, steady_state_name], reload=False)
        self.exp.get_data_object(name).df = np.abs(self.exp.get_df(name))

        time_name = 't'
        column_names = self.exp.get_data_object(name).get_column_names()

        if start_frame_intervals is None or end_frame_intervals is None:
            dx = 0.25
            start_frame_intervals = np.array(np.arange(0, 3.5, dx)*60*100, dtype=int)
            end_frame_intervals = np.array(start_frame_intervals + 1000, dtype=int)

        dtheta = np.pi / 25.
        bins = np.arange(0, np.pi, dtheta)

        if n is None:
            n = int(np.floor(np.sqrt(len(column_names))))
            m = int(np.ceil(len(column_names) / n))

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(name))
        top, bottom, left, right, wspace, hspace = self.get_adjust(adjust)
        fig, ax = plotter.create_plot(
            figsize=(4 * m, 4 * n), nrows=n, ncols=m,
            top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)

        plotter_steadystate = Plotter(root=self.exp.root, obj=self.exp.get_data_object(steady_state_name))
        for k, column_name in enumerate(column_names):
            title = self.get_title(column_name, title_option)

            hist_name = self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                                  end_frame_intervals=end_frame_intervals, bins=bins, fps=fps,
                                                  index_name=time_name, column_to_hist=column_name, replace=True)

            i = int(np.floor(k / m))
            j = k % m
            if isinstance(ax, np.ndarray):
                if len(ax.shape) == 1:
                    ax0 = ax[k]
                else:
                    ax0 = ax[i, j]
            else:
                ax0 = ax
            ax0.set_ylim(0, 0.5)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
            plotter.plot(xlabel=r'$\theta$ (rad)', ylabel='PDF', title=title, normed=2, preplot=(fig, ax0),
                         display_legend=False)
            plotter_steadystate.plot(
                xlabel=r'$\theta$ (rad)', ylabel='PDF', title=title, normed=2, preplot=(fig, ax0),
                c='navy', marker=None, alpha=0.5, label='Exp. steady-state', display_legend=False)

            if k == 0:
                plotter.draw_legend(ax0, ncol=2)

        fig_name = name + '_hist_pretty'
        plotter.save(fig, name=fig_name)

        self.exp.remove_object(name)

    @staticmethod
    def get_adjust(adjust):
        if adjust is None:
            adjust = {}

        if 'top' in adjust:
            top = adjust['top']
        else:
            top = 0.98

        if 'bottom' in adjust:
            bottom = adjust['bottom']
        else:
            bottom = .12

        if 'left' in adjust:
            left = adjust['left']
        else:
            left = .1

        if 'right' in adjust:
            right = adjust['right']
        else:
            right = None

        if 'wspace' in adjust:
            wspace = adjust['wspace']
        else:
            wspace = None

        if 'hspace' in adjust:
            hspace = adjust['hspace']
        else:
            hspace = None

        return top, bottom, left, right, wspace, hspace

    @staticmethod
    def get_title(column_name, display_title):
        list_para = list(column_name.split(','))
        list_para[0] = list_para[0][1:]
        list_para[-1] = list_para[-1][:-1]
        if len(list_para[1]) == 0:
            list_para.pop()
        list_para = np.array(list_para, dtype=float)

        if display_title is None:
            title = ''
        elif isinstance(display_title[1], int):
            title = display_title[0] + ' = ' + str(list_para[display_title[1]])
        else:
            paras = tuple(list_para[display_title[1]])
            title = display_title[0] + ' = ' + str(paras)
        return title

    def plot_var_model_pretty(
            self, name, n=None, m=None, title_option=None, adjust=None, suff=None, plot_fisher=False):

        if suff is not None:
            name += '_'+suff
        name_exp_variance = 'food_direction_error_var_evol_around_first_attachment'
        self.exp.load([name, name_exp_variance], reload=False)
        if plot_fisher is True:
            name_exp_fisher_info = 'fisher_info_evol_around_first_attachment'
            self.exp.load(name_exp_fisher_info, reload=False)
        else:
            name_exp_fisher_info = None

        time_name = 't'
        column_names = self.exp.get_data_object(name).get_column_names()

        dx = 0.05
        dx2 = 0.01
        start_frame_intervals = np.arange(0, 3., dx2)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        if n is None:
            n = int(np.floor(np.sqrt(len(column_names))))
            m = int(np.ceil(len(column_names) / n))

        top, bottom, left, right, wspace, hspace = self.get_adjust(adjust)

        plotter_exp_variance = Plotter(
            root=self.exp.root, obj=self.exp.get_data_object(name_exp_variance), category=self.category)
        fig, ax = plotter_exp_variance.create_plot(
            figsize=(4 * m, 4 * n), nrows=n, ncols=m,
            top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)

        if plot_fisher is True:
            plotter_exp_fisher_info = Plotter(
                root=self.exp.root, obj=self.exp.get_data_object(name_exp_fisher_info), category=self.category)
            fig2, ax2 = plotter_exp_fisher_info.create_plot(
                figsize=(4 * m, 4 * n), nrows=n, ncols=m,
                top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)
        else:
            fig2 = fig
            ax2 = ax

        for k, column_name in enumerate(column_names):

            var_name = self.exp.variance_evolution(name_to_var=name,  start_index_intervals=start_frame_intervals,
                                                   end_index_intervals=end_frame_intervals, index_name=time_name,
                                                   column_to_var=column_name, replace=True)
            self.exp.get_df(var_name).index = self.exp.get_index(var_name)/100.

            i = int(np.floor(k / m))
            j = k % m
            if isinstance(ax, np.ndarray):
                if len(ax.shape) == 1:
                    ax0 = ax[k]
                    ax02 = ax2[k]
                else:
                    ax0 = ax[i, j]
                    ax02 = ax2[i, j]
            else:
                ax0 = ax
                ax02 = ax2

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(var_name))

            title = self.get_title(column_name, title_option)

            c = 'darkorange'
            plotter_exp_variance.plot(
                xlabel='Time (s)', ylabel=r'$\sigma$ (rad$^2$)', preplot=(fig, ax0),
                label='Experiment', marker='', title=title)
            plotter_exp_variance.plot_fit(
                typ='exp', preplot=(fig, ax0), window=[90, 400], cst=(-0.01, .1, .1), label='Exponential fit')
            plotter.plot(
                xlabel='Time (s)', ylabel=r'$\sigma$ (rad$^2$)',
                preplot=(fig, ax0), label='Model', c=c, marker='', title=title)

            ax0.legend()
            plotter.draw_vertical_line(ax0)
            if plot_fisher is True:
                plotter_exp_variance.plot(
                    xlabel='Time (s)', ylabel=r'Fisher information (rad$^{-2}$)',
                    fct_y=lambda a: 1 / a, preplot=(fig2, ax02),
                    label='Experiment', marker='', title=title)
                plotter.plot(xlabel='Time (s)', ylabel=r'Fisher information (rad$^{-2}$)', fct_y=lambda a: 1 / a,
                             preplot=(fig2, ax02), label='Model', c=('%s' % c), marker='', title=title)

                ax02.legend()
                plotter.draw_vertical_line(ax02)

        plotter_exp_variance.save(fig, name=name+'_var_pretty')
        if plot_fisher is True:
            plotter_exp_variance.save(fig2, name=name+'_fisher_info_pretty')

    def compare_norm_vonmises(self):
        list_var = [0.1, 0.25, 0.5, 1, 2]
        x = np.arange(-np.pi, np.pi, 0.01)
        x2 = np.arange(-np.pi, np.pi, 0.1)
        plotter = BasePlotters()
        cols = plotter.color_object.create_cmap('hot', list_var)
        fig, ax = plotter.create_plot()
        for var in list_var:
            ax.plot(x, scs.norm.pdf(x, scale=np.sqrt(var)), label=r'$s^2$=%.2f' % var, c=cols[str(var)])
        for var in list_var:
            ax.plot(
                x2, scs.vonmises.pdf(x2, kappa=1/var),
                'o', c=cols[str(var)], ms=3.5, label=r'$1/\kappa$=%.2f' % var)
        ax.legend()
        ax.set_xlabel('x (rad)')
        ax.set_ylabel('pdf')

        address = '%s%s/Plots/%s.png' % (self.exp.root, self.category, 'gaussian_vs_vonmises')
        fig.savefig(address)
