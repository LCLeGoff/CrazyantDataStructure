import numpy as np
import pandas as pd
from scipy import stats as scs
from scipy import optimize as scopt

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from AnalyseClasses.Models.UOWrappedModels import UOWrappedSimpleModel
from DataStructure.VariableNames import id_exp_name, id_frame_name, id_ant_name
from Scripts.root import root
from Tools.MiscellaneousTools import Geometry as Geo
from Tools.MiscellaneousTools.ArrayManipulation import get_index_interval_containing
from Tools.Plotter.BasePlotters import BasePlotters
from Tools.Plotter.Plotter import Plotter

import matplotlib as mlt
mlt.rcParams['axes.titlesize'] = 15
mlt.rcParams['axes.labelsize'] = 13


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
            title = self.get_label(column_name, title_option)

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

    def plot_hist_fit_model_pretty(
            self, name, para, n=None, m=None,
            start_frame_intervals=None, end_frame_intervals=None, suff=None, adjust=None, title_option=None):

        if suff is not None:
            name += '_'+suff

        exp_name = 'food_direction_error'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        self.exp.load([name, exp_name, first_attachment_name], reload=False)

        self.change_first_frame(exp_name, first_attachment_name)
        self.exp.operation(name, np.abs)
        self.exp.operation(exp_name, np.abs)

        if start_frame_intervals is None or end_frame_intervals is None:
            dx = 0.25
            start_frame_intervals = np.array(np.arange(0, 2, dx)*60*100, dtype=int)
            end_frame_intervals = np.array(start_frame_intervals + 1500, dtype=int)

        dtheta = np.pi / 25.
        bins = np.arange(0, np.pi, dtheta)
        x = (bins[1:]+bins[:-1])/2.

        lg = len(start_frame_intervals)
        if n is None:
            n = int(np.floor(np.sqrt(lg)))
            m = int(np.ceil(lg / n))

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(name))
        top, bottom, left, right, wspace, hspace = self.get_adjust(adjust)
        fig, ax = plotter.create_plot(
            figsize=(4 * m, 4 * n), nrows=n, ncols=m,
            top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)
        title = self.get_label(para, option=title_option)
        fig.suptitle(title, fontsize=20)

        temp_name = 'temp'
        temp_name_exp = 'temp_exp'
        cols = plotter.color_object.create_cmap('hot', range(len(start_frame_intervals)))
        for k in range(lg):

            j0 = int(np.floor(k / m))
            j1 = k % m
            t0 = start_frame_intervals[k]
            t1 = end_frame_intervals[k]

            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=[t0], end_frame_intervals=[t1],
                                      bins=bins, fps=100., index_name='t', column_to_hist=para,
                                      result_name=temp_name, category=self.category, replace=True)

            self.exp.hist1d_evolution(name_to_hist=exp_name, start_frame_intervals=[t0],
                                      end_frame_intervals=[t1], bins=bins,
                                      result_name=temp_name_exp, category=self.category, replace=True)

            if isinstance(ax, np.ndarray):
                if len(ax.shape) == 1:
                    ax0 = ax[k]
                else:
                    ax0 = ax[j0, j1]
            else:
                ax0 = ax
            ax0.set_ylim(0, 0.5)

            c = cols[str(k)]

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(temp_name_exp))
            plotter.plot(preplot=(fig, ax[j0, j1]), marker=None, c='grey',
                         display_legend=False, label='Experimental', normed=2)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(temp_name))
            plotter.plot(xlabel=r'$\theta$ (rad)', ylabel='PDF', title=r't$\in$[%i, %i]s' % (t0/100, t1/100),
                         normed=2, preplot=(fig, ax0), label='model',
                         display_legend=False, c=c, ls='')

            y = self.exp.get_df(temp_name).values.ravel()
            s = np.sum(y)
            y = y/s / dtheta/2.
            popt, _ = scopt.curve_fit(self._uniform_vonmises_dist, x, y, p0=[0.2, 2], bounds=[(0, 0), (1, np.inf)])
            q = round(popt[0], 3)
            kappa = round(popt[1], 3)
            y_fit = self._uniform_vonmises_dist(x, q, kappa)

            ax[j0, j1].plot(x, y_fit, c=cols[str(k)], label=r'q=%.3f, $\kappa$=%.3f' % (q, kappa))
            ax[j0, j1].set_ylim(0, 0.5)
            plotter.draw_legend(ax[j0, j1])

            if k == 0:
                plotter.draw_legend(ax0, ncol=2)

        fig_name = name + '_hist_fit_pretty_'+para
        plotter.save(fig, name=fig_name)

        self.exp.remove_object(name)
        self.exp.remove_object(exp_name)

    @staticmethod
    def _uniform_vonmises_dist(x, q, kappa):
        y = q*scs.vonmises.pdf(x, kappa)+(1-q)/(2*np.pi)
        return y

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
    def get_label(column_name, option=None):
        list_para = list(column_name.split(','))
        list_para[0] = list_para[0][1:]
        list_para[-1] = list_para[-1][:-1]
        if len(list_para[1]) == 0:
            list_para.pop()
        list_para = np.array(list_para, dtype=float)

        if option is None:
            title = str(column_name)
        elif isinstance(option[1], int):
            title = option[0] + ' = ' + str(list_para[option[1]])
        else:
            paras = tuple(list_para[option[1]])
            title = option[0] + ' = ' + str(paras)
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

            title = self.get_label(column_name, title_option)

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

    def compute_path_efficiency(self, name_model, suff=None, redo=False, redo_hist=False, label_option=None):

        window = 10

        if suff is not None:
            name_model += '_'+suff

        exp_name = 'w' + str(window) + 's_food_path_efficiency_hist'
        result_name = 'path_efficiency_%s' % name_model

        if redo:
            self.exp.load(name_model, reload=False)

            df = self.exp.get_df(name_model).copy()
            fps = int(100/float(df.index.get_level_values('t')[1]))

            df_cos = np.cos(df).groupby(id_exp_name).rolling(window=window*fps, center=True).sum()
            df_sin = np.sin(df).groupby(id_exp_name).rolling(window=window*fps, center=True).sum()
            df_cos.index = df.index
            df_sin.index = df.index
            df_dist = np.sqrt(df_cos**2+df_sin**2)

            df_efficiency = np.around(df_dist/window/fps, 3)

            self.exp.add_new_dataset_from_df(df=df_efficiency, name=result_name, category=self.category,
                                             label='', description='')

            self.exp.write(result_name)

        self.exp.load(exp_name)
        plotter = Plotter(self.exp.root, self.exp.get_data_object(exp_name))
        fig, ax = plotter.plot(normed=True, c='navy')

        bins = np.arange(0, 1, 0.05)
        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        labels = []
        for column_name in self.exp.get_columns(hist_name):
            labels.append(self.get_label(column_name=column_name, option=label_option))

        plotter = Plotter(self.exp.root, self.exp.get_data_object(hist_name))
        plotter.plot(preplot=(fig, ax), normed=True, label=labels)
        plotter.save(fig)

    def compute_plot_path_efficiency(self, name_model, para, suff=None, redo=False, label_option=None):

        if suff is not None:
            name_model += '_' + suff

        name_eff = 'path_efficiency_%s' % name_model

        self.exp.load([name_model, name_eff])

        temp_name = '%s_%s' % (name_model, name_eff)
        result_name = '%s_hist_evol_%s_%s' % (name_model, name_eff, para)

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        d_eff = 0.1
        start_eff_intervals = np.around(np.arange(0, 1, d_eff), 1)
        end_eff_intervals = np.around(start_eff_intervals+d_eff, 1)

        label = 'Object orientation distribution over the path efficiency (rad)'
        description = 'Object orientation distribution over the path efficiency (rad)' \
                      ' of the model %s of parameter %s' % (name_model, para)

        if redo:
            self.exp.load([name_model, name_eff])

            self._add_path_efficiency_index(name_model, para, name_eff, temp_name, w=16)
            self.exp.operation(temp_name, lambda a: np.abs(a))

            self.exp.hist1d_evolution(name_to_hist=temp_name, start_frame_intervals=start_eff_intervals,
                                      end_frame_intervals=end_eff_intervals, bins=bins, index_name=name_eff,
                                      result_name=result_name, category=self.category,
                                      label=label, description=description)

            self.exp.remove_object(name_model)
            self.exp.remove_object(name_eff)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        labels = []
        for column_name in self.exp.get_columns(result_name):
            labels.append(self.get_label(column_name=column_name, option=label_option))
        fig, ax = plotter.plot(xlabel=r'$\theta_{error}$ (rad)', ylabel='PDF', normed=2,
                               title='', label=labels)
        ax.set_ylim((0, 0.6))
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def _add_path_efficiency_index(self, name, para, name_eff, temp_name, w):

        index_error = self.exp.get_index(name).copy()
        index_exp = index_error.get_level_values(id_exp_name)
        index_frame = index_error.get_level_values('t')

        df_eff = pd.DataFrame(self.exp.get_df(name_eff)[para], columns=[para])
        df_eff.reset_index(inplace=True)
        df_eff['t'] += 50*w
        df_eff.set_index([id_exp_name, 't'], inplace=True)
        df_eff = df_eff.reindex(index_error)

        index_discrim = np.around(df_eff.loc[index_error].values.ravel(), 3)
        mask = np.where(~np.isnan(index_discrim))[0]
        index1 = list(zip(index_exp[mask], index_frame[mask]))
        index2 = list(zip(index_exp[mask], index_frame[mask], index_discrim[mask]))
        index1 = pd.MultiIndex.from_tuples(index1, names=[id_exp_name, 't'])
        index2 = pd.MultiIndex.from_tuples(index2, names=[id_exp_name, 't', name_eff])
        df = pd.DataFrame(self.exp.get_df(name)[para], columns=[para])
        df = df.reindex(index1)
        df.index = index2
        self.exp.add_new_dataset_from_df(df=df, name=temp_name, replace=True)

    def _add_path_efficiency_index_for_exp(self, name, discrim_name, temp_name, w):
        index_error = self.exp.get_index(name).copy()
        index_exp = index_error.get_level_values(id_exp_name)
        index_frame = index_error.get_level_values(id_frame_name)

        self.exp.get_df(discrim_name).reset_index(inplace=True)
        self.exp.get_df(discrim_name)[id_frame_name] += 50*w
        self.exp.get_df(discrim_name).set_index([id_exp_name, id_frame_name], inplace=True)
        self.exp.change_df(discrim_name, self.exp.get_df(discrim_name).reindex(index_error))

        index_discrim = np.around(self.exp.get_df(discrim_name).loc[index_error].values.ravel(), 3)

        mask = np.where(~np.isnan(index_discrim))[0]
        index1 = list(zip(index_exp[mask], index_frame[mask]))
        index2 = list(zip(index_exp[mask], index_frame[mask], index_discrim[mask]))
        index1 = pd.MultiIndex.from_tuples(index1, names=[id_exp_name, id_frame_name])
        index2 = pd.MultiIndex.from_tuples(index2, names=[id_exp_name, id_frame_name, discrim_name])

        df = self.exp.get_df(name).copy()
        df = df.reindex(index1)
        df.index = index2

        self.exp.add_new_dataset_from_df(df=df, name=temp_name, replace=True)

    def compute_plot_path_efficiency_fit(self, name_model, para, suff=None, title_option=None):
        if suff is not None:
            name_model += '_' + suff

        name_eff = 'path_efficiency_%s' % name_model
        name_exp = 'food_direction_error'
        name_eff_exp = 'w16s_food_path_efficiency_resolution1pc'

        self.exp.load([name_model, name_eff, name_exp, name_eff_exp])

        temp_name = 'temp'
        temp_exp_name = 'temp_exp'
        result_name = '%s_hist_evol_%s_%s_fit' % (name_model, name_eff, para)

        start_eff_intervals = np.around([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 2)
        end_eff_intervals = np.around([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 2)

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)
        x = (bins[1:]+bins[:-1])/2.

        self._add_path_efficiency_index(name_model, para, name_eff, temp_name, w=16)
        self._add_path_efficiency_index_for_exp(name_exp, name_eff_exp, temp_exp_name, w=16)

        self.exp.operation(temp_name, np.abs)
        self.exp.operation(temp_exp_name, np.abs)

        plotter = BasePlotters()
        cols = plotter.color_object.create_cmap('hot', range(len(start_eff_intervals)))
        fig, ax = plotter.create_plot(
            figsize=(8, 8), nrows=3, ncols=3,  left=0.08, bottom=0.06, top=0.96, hspace=0.4, wspace=0.3)

        title = self.get_label(para, option=title_option)
        fig.suptitle(title)

        for ii in range(len(start_eff_intervals)):
            j0 = int(ii/3)
            j1 = ii-j0*3

            c = cols[str(ii)]
            eff0 = start_eff_intervals[ii]
            eff1 = end_eff_intervals[ii]

            hist_name = self.exp.hist1d_evolution(name_to_hist=temp_name, start_frame_intervals=[eff0],
                                                  end_frame_intervals=[eff1], bins=bins, index_name=name_eff,
                                                  category=self.category, replace=True)

            hist_exp_name = self.exp.hist1d_evolution(name_to_hist=temp_exp_name, start_frame_intervals=[eff0],
                                                      end_frame_intervals=[eff1], bins=bins,
                                                      category=self.category, replace=True)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_exp_name))
            plotter.plot(
                preplot=(fig, ax[j0, j1]), xlabel=r'$\theta$ (rad)', ylabel='PDF',
                title=r'Eff.$\in$[%.2f, %.2f]' % (eff0, eff1), label='Experiment', normed=2,
                marker=None, c='navy', display_legend=False
                )

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
            plotter.plot(
                preplot=(fig, ax[j0, j1]), xlabel=r'$\theta$ (rad)', ylabel='PDF', ls='',
                title=r'Eff.$\in$[%.2f, %.2f]' % (eff0, eff1), c=c,
                display_legend=False, label='model', normed=2)

            y = self.exp.get_df(hist_name).values.ravel()
            mask = np.where(~np.isnan(y))[0]
            if len(mask) > 0:
                y = y[mask]
                x2 = x[mask]
                s = np.sum(y)
                y = y/s / dtheta/2.
                popt, _ = scopt.curve_fit(self._uniform_vonmises_dist, x2, y, p0=[0.2, 2.], bounds=(0, [1, np.inf]))
                q = round(popt[0], 2)
                kappa = round(popt[1], 2)

                y_fit = self._uniform_vonmises_dist(x2, q, kappa)

                ax[j0, j1].plot(x, y_fit, c=c, label=r'q=%.2f, $\kappa$=%.2f' % (q, kappa))
            ax[j0, j1].set_ylim(0, 1)
            plotter.draw_legend(ax[j0, j1])

        plotter.save(fig, name=result_name)
        self.exp.remove_object(name_exp)

    def _get_rolling_prop(self, discrim_name, para, w, temp_name):

        df_out = self.exp.get_df(discrim_name)[para].copy()
        df_out[df_out == 2] = 0
        df_out = df_out.rolling(window=w, min_periods=2).mean()

        df_in = self.exp.get_df(discrim_name)[para].copy()
        df_in[df_in == 1] = 0
        df_in[df_in == 2] = 1
        df_in = df_in.rolling(window=w, min_periods=2).mean()

        df = df_out / (df_in + df_out)

        self.exp.add_new_dataset_from_df(df=df, name=temp_name, replace=True)

    def _get_rolling_prop4exp(self, discrim_name, w):
        self.exp.load(['outside_' + discrim_name, 'inside_' + discrim_name])

        df_out = self.exp.get_df('outside_' + discrim_name).copy()
        df_out.reset_index(inplace=True)
        df_out.drop(columns=id_ant_name, inplace=True)
        df_out.set_index([id_exp_name, id_frame_name], inplace=True)
        df_out = df_out.rolling(window=w * 100, min_periods=100).mean() * 100

        df_in = self.exp.get_df('inside_' + discrim_name).copy()
        df_in.reset_index(inplace=True)
        df_in.drop(columns=id_ant_name, inplace=True)
        df_in.set_index([id_exp_name, id_frame_name], inplace=True)
        df_in = df_in.rolling(window=w * 100, min_periods=100).mean() * 100
        df_in.columns = ['outside_' + discrim_name]
        df = df_out / (df_in + df_out)

        self.exp.change_df('outside_' + discrim_name, df)

    def _add_attachment_index(self, name, discrim_name, para, temp_name):

        index_error = self.exp.get_index(name).copy()
        index_exp = index_error.get_level_values(id_exp_name)
        index_frame = index_error.get_level_values('t')

        df_eff = pd.DataFrame(self.exp.get_df(discrim_name)[para], columns=[para])
        index_discrim = np.around(df_eff.loc[index_error].values.ravel(), 6)

        mask = np.where(~np.isnan(index_discrim))[0]
        index1 = list(zip(index_exp[mask], index_frame[mask]))
        index2 = list(zip(index_exp[mask], index_frame[mask], index_discrim[mask]))
        index1 = pd.MultiIndex.from_tuples(index1, names=[id_exp_name, 't'])
        index2 = pd.MultiIndex.from_tuples(index2, names=[id_exp_name, 't', discrim_name])

        df = pd.DataFrame(self.exp.get_df(name)[para], columns=[para])
        df = df.reindex(index1)
        df.index = index2

        self.exp.add_new_dataset_from_df(df=df, name=temp_name, replace=True)

    def _add_attachment_index_for_exp(self, name, discrim_name, temp_name):
        index_error = self.exp.get_index(name).copy()
        index_exp = index_error.get_level_values(id_exp_name)
        index_frame = index_error.get_level_values(id_frame_name)

        index_discrim = np.around(self.exp.get_df(discrim_name).loc[index_error].values.ravel(), 6)

        mask = np.where(~np.isnan(index_discrim))[0]
        index1 = list(zip(index_exp[mask], index_frame[mask]))
        index2 = list(zip(index_exp[mask], index_frame[mask], index_discrim[mask]))
        index1 = pd.MultiIndex.from_tuples(index1, names=[id_exp_name, id_frame_name])
        index2 = pd.MultiIndex.from_tuples(index2, names=[id_exp_name, id_frame_name, discrim_name])

        df = self.exp.get_df(name).copy()
        df = df.reindex(index1)
        df.index = index2

        self.exp.add_new_dataset_from_df(df=df, name=temp_name, replace=True)

    def compute_plot_attachment_prop_fit(self, name_model, para, suff=None, title_option=None):
        if suff is not None:
            name_model += '_' + suff

        name_attachments = name_model + '_attachments'

        name_exp = 'food_direction_error'
        name_eff_exp = 'attachments'
        last_frame_name = 'food_exit_frames'

        self.exp.load([name_model, name_attachments, name_exp, name_eff_exp, last_frame_name])
        self.cut_last_frames_for_indexed_by_exp_frame_indexed(name_exp, last_frame_name)

        temp_name = 'temp'
        temp_exp_name = 'temp_exp'
        result_name = '%s_hist_evol_%s_%s_fit' % (name_model, 'attachment_prop', para)

        start_eff_intervals = np.around([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 2)
        end_eff_intervals = np.around([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 2)

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)
        x = (bins[1:]+bins[:-1])/2.

        self._get_rolling_prop(name_attachments, para, 10, temp_name)
        self._add_attachment_index(name_model, temp_name, para, temp_name)

        self._get_rolling_prop4exp(name_eff_exp, 10)
        self._add_attachment_index_for_exp(name_exp, 'outside_'+name_eff_exp, temp_exp_name)

        self.exp.operation(temp_name, np.abs)
        self.exp.operation(temp_exp_name, np.abs)

        plotter = BasePlotters()
        cols = plotter.color_object.create_cmap('hot', range(len(start_eff_intervals)))
        fig, ax = plotter.create_plot(
            figsize=(8, 8), nrows=3, ncols=3,  left=0.08, bottom=0.06, top=0.96, hspace=0.4, wspace=0.3)

        title = self.get_label(para, option=title_option)
        fig.suptitle(title)

        for ii in range(len(start_eff_intervals)):
            j0 = int(ii/3)
            j1 = ii-j0*3

            c = cols[str(ii)]
            eff0 = start_eff_intervals[ii]
            eff1 = end_eff_intervals[ii]

            hist_name = self.exp.hist1d_evolution(name_to_hist=temp_name, start_frame_intervals=[eff0],
                                                  end_frame_intervals=[eff1], bins=bins,
                                                  category=self.category, replace=True)

            hist_exp_name = self.exp.hist1d_evolution(name_to_hist=temp_exp_name, start_frame_intervals=[eff0],
                                                      end_frame_intervals=[eff1], bins=bins,
                                                      category=self.category, replace=True)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_exp_name))
            plotter.plot(
                preplot=(fig, ax[j0, j1]), xlabel=r'$\theta$ (rad)', ylabel='PDF',
                title=r'Eff.$\in$[%.2f, %.2f]' % (eff0, eff1), label='Experiment', normed=2,
                marker=None, c='navy', display_legend=False
                )

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
            plotter.plot(
                preplot=(fig, ax[j0, j1]), xlabel=r'$\theta$ (rad)', ylabel='PDF', ls='',
                title=r'Eff.$\in$[%.2f, %.2f]' % (eff0, eff1), c=c,
                display_legend=False, label='model', normed=2)

            y = self.exp.get_df(hist_name).values.ravel()
            mask = np.where(~np.isnan(y))[0]
            if len(mask) > 0:
                y = y[mask]
                x2 = x[mask]
                s = np.sum(y)
                if s > 0:
                    y = y/s / dtheta/2.
                popt, _ = scopt.curve_fit(self._uniform_vonmises_dist, x2, y, p0=[0.2, 2.], bounds=(0, [1, np.inf]))
                q = round(popt[0], 2)
                kappa = round(popt[1], 2)

                y_fit = self._uniform_vonmises_dist(x2, q, kappa)

                ax[j0, j1].plot(x, y_fit, c=c, label=r'q=%.2f, $\kappa$=%.2f' % (q, kappa))
            ax[j0, j1].set_ylim(0, 1)
            plotter.draw_legend(ax[j0, j1])

        plotter.save(fig, name=result_name)
        self.exp.remove_object(name_exp)
