import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
import pylab as pb

import Tools.MiscellaneousTools.Geometry as Geo
import scipy.stats as scs
import scipy.optimize as scopt

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from Tools.MiscellaneousTools.FoodFisherInformation import compute_fisher_information_uniform_von_mises
from Tools.Plotter.BasePlotters import BasePlotters
from DataStructure.VariableNames import id_exp_name, id_frame_name, id_ant_name
from Tools.MiscellaneousTools import Fits
from Tools.Plotter.Plotter import Plotter


class AnalyseFoodVeracity(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'FoodVeracity'

    def compute_food_direction_error(self, redo=False, redo_hist=False, redo_plot_indiv=False):

        food_exit_angle_name = 'food_exit_angle'
        food_phi_name = 'food_velocity_phi'
        result_name = 'food_direction_error'

        dtheta = np.pi/25.
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)

        result_label = 'Food direction error (rad)'
        result_description = 'Angle between the food velocity and the food-exit vector,' \
                             'which gives in radian how much the food is not going in the good direction'
        if redo:
            self.exp.load([food_exit_angle_name, food_phi_name])

            tab = Geo.angle_distance(self.exp.get_df(food_exit_angle_name)[food_exit_angle_name],
                                     self.exp.get_df(food_phi_name)[food_phi_name])

            self.exp.add_copy1d(name_to_copy=food_phi_name, copy_name=result_name, category=self.category,
                                label=result_label, description=result_description)

            self.exp.change_values(result_name, np.around(tab, 6))

            self.exp.write(result_name)

        else:
            self.exp.load(result_name)

        self.__plot_indiv(result_name, redo, redo_plot_indiv)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter2 = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig2, ax2 = plotter2.plot(xlabel='Food direction error', ylabel='PDF', normed=True)
        plotter2.save(fig2)

    def compute_food_direction_error_hist_evol(self, redo=False):
        name = 'food_direction_error'
        result_name = name+'_hist_evol'
        init_frame_name = 'food_first_frame'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        dx = 0.25
        start_frame_intervals = np.array(np.arange(0, 3.5, dx)*60*100, dtype=int)
        end_frame_intervals = np.array(start_frame_intervals + dx*60*100*2, dtype=int)

        hist_label = 'Food direction error distribution over time (rad)'
        hist_description = 'Histogram of the angle between the food velocity and the food-exit vector,' \
                           'which gives in radian how much the food is not going in the good direction (rad)'

        if redo:
            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            self.exp.operation(name, np.abs)
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                      end_frame_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category,
                                      label=hist_label, description=hist_description)
            self.exp.remove_object(name)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta_{error}$ (rad)', ylabel='PDF', normed=True, label_suffix='s',
                               title='')
        ax.set_ylim((0, 1))
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(yscale='log', xscale='log',
                               xlabel=r'$\theta_{error}$ (rad)', ylabel='PDF', label_suffix='s', normed=True,
                               title='')
        plotter.save(fig, suffix='power')

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(
            yscale='log', xlabel=r'$\theta_{error}$ (rad)', ylabel='PDF', label_suffix='s', normed=True, title='')
        x = self.exp.get_index(result_name)
        lamb = 0.77

        ax.plot(x, lamb*np.exp(-lamb*x), ls='--', c='k', label=str(lamb)+' exp(-'+str(lamb)+r'$\varphi$)')
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig, suffix='exp')

        column = self.exp.get_columns(result_name)[-2]
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name=column)
        fig, ax = plotter.plot(yscale='log', xlabel=r'$\theta_{error}$ (rad)', ylabel='PDF', label_suffix='s',
                               normed=True, label=r'Variance at times t$\in$' + str(column) + ' s')
        plotter.plot_fit(preplot=(fig, ax), normed=True)
        plotter.save(fig, suffix='steady_state')

        self.exp.get_df(result_name).index = self.exp.get_df(result_name).index**2
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(yscale='log', xlabel=r'$\theta_{error}$ (rad)', ylabel='PDF',
                               label_suffix='s', normed=True)
        plotter.save(fig, suffix='gauss')

    def compute_food_direction_error_variance_evol(self, redo=False):
        name = 'food_direction_error'
        result_name = name + '_var_evol'
        init_frame_name = 'food_first_frame'

        dx = 0.1
        dx2 = 0.01
        start_frame_intervals = np.arange(0, 3.5, dx2)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        label = 'Variance of the food direction error distribution over time'
        description = 'Variance of the angle between the food velocity and the food-exit vector,' \
                      'which gives in radian how much the food is not going in the good direction'

        if redo:
            self.exp.load([init_frame_name, name])

            self.change_first_frame(name, init_frame_name)

            self.exp.variance_evolution(name_to_var=name, start_index_intervals=start_frame_intervals,
                                        end_index_intervals=end_frame_intervals,
                                        category=self.category, result_name=result_name,
                                        label=label, description=description)

            self.exp.write(result_name)
            self.exp.remove_object(name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(
            xlabel='Time (s)', ylabel=r'Variance $\sigma^2$',
            label_suffix='s', label=r'$\sigma^2$', title='', marker='')
        plotter.plot_fit(typ='exp', preplot=(fig, ax), window=[0, 400], cst=(-0.01, .1, .1))
        plotter.save(fig)

    def compute_food_direction_error_hist_evol_around_first_attachment(self, redo):
        name = 'food_direction_error'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        result_name = name+'_hist_evol_around_first_attachment'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        dx = 0.25
        start_frame_intervals = np.arange(-1, 2.1, dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        result_label = 'Food direction error histogram evolution over time'
        result_description = 'Evolution over time of the histogram of food error direction,negative times (s)' \
                             ' correspond to periods before the first attachment of an outside ant'

        self.exp.load(name)
        self.change_first_frame(name, first_attachment_name)
        self.exp.operation(name, np.abs)
        if redo:

            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                      end_frame_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category,
                                      label=result_label, description=result_description)

            self.exp.write(result_name)
            self.exp.remove_object(name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta$ (rad)', ylabel='PDF', normed=2, label_suffix=' s', title='')
        plotter.draw_legend(ax=ax, ncol=2)
        ax.set_ylim((0, 0.5))
        plotter.save(fig)

        temp_name = 'temp'
        self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=[45 * 100],
                                  end_frame_intervals=[150 * 100], bins=bins,
                                  result_name=temp_name, category=self.category,
                                  label=result_label, description=result_description)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta$ (rad)', ylabel='PDF', normed=True, label_suffix=' s', title='')
        plotter2 = Plotter(root=self.exp.root, obj=self.exp.get_data_object(temp_name))
        plotter2.plot_fit(preplot=(fig, ax), normed=True, c='navy', typ='cst center gauss', label_suff='gauss fit')
        plotter2.plot_fit(preplot=(fig, ax), normed=True, c='darkcyan', typ='cst vonmises', label_suff='mises fit')
        plotter.draw_legend(ax=ax, ncol=2)
        ax.set_ylim((0, 1))
        plotter.save(fig, suffix='fit')

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(
            yscale='log', xlabel=r'$\theta$ (rad)', ylabel='PDF', label_suffix='s', normed=True, title='')
        x = self.exp.get_index(result_name)
        lamb = 0.77

        ax.plot(x, lamb*np.exp(-lamb*x), ls='--', c='k', label=str(lamb)+' exp(-'+str(lamb)+r'$\varphi$)')
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig, suffix='exp')

        self.exp.get_df(result_name).index = self.exp.get_df(result_name).index**2
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(yscale='log', xlabel=r'$\theta$ (rad)', ylabel='PDF', normed=True, label_suffix='s')
        plotter.save(fig, suffix='gauss')

    def compute_food_direction_error_hist_evol_around_first_attachment_steady_state_fit(self):
        name = 'food_direction_error'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        result_name = name+'_hist_evol_around_first_attachment'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        result_label = 'Food direction error steady-state distribution'
        result_description = 'Histogram of food error direction after 45s, time 0 is the first outside attachment time'

        self.exp.load(name)
        self.change_first_frame(name, first_attachment_name)
        self.exp.operation(name, np.abs)

        steady_state_name = 'food_direction_error_hist_steady_state'
        self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=[60 * 100],
                                  end_frame_intervals=[120 * 100], bins=bins,
                                  result_name=steady_state_name, category=self.category,
                                  label=result_label, description=result_description)
        self.exp.write(steady_state_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(steady_state_name))
        fig, ax = plotter.plot(
            xlabel=r'$\theta$ (rad)', ylabel='PDF', ls='',
            normed=2, label_suffix=' s', title='', label='Experimental steady-state')

        plotter.plot_fit(preplot=(fig, ax), typ='uniform vonmises', normed=2, label_suff='fit')
        ax.set_ylim(0, 0.4)

        plotter.draw_legend(ax)
        plotter.save(fig, name=result_name+'_steady_state_fit')

    def compute_food_direction_error_hist_evol_around_first_attachment_evol_fit(self):
        name = 'food_direction_error'
        first_attachment_name = 'first_attachment_time_of_outside_ant'
        result_name = name+'_hist_evol_around_first_attachment'

        dx = 0.25
        start_frame_intervals = np.arange(-.25, 2., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)
        x = (bins[1:]+bins[:-1])/2.

        self.exp.load(name)
        self.change_first_frame(name, first_attachment_name)
        self.exp.operation(name, np.abs)

        temp_name = 'temp'

        # ts = np.array([0, 200, 1500, 3000, 4500])
        # dt = 500
        plotter = BasePlotters()
        cols = plotter.color_object.create_cmap('hot', range(len(start_frame_intervals)))
        fig, ax = plotter.create_plot(
            figsize=(8, 8), nrows=3, ncols=3,  left=0.08, bottom=0.06, top=0.96, hspace=0.4, wspace=0.3)
        for ii in range(len(start_frame_intervals)):
            j0 = int(ii/3)
            j1 = ii-j0*3
            t0 = start_frame_intervals[ii]
            t1 = end_frame_intervals[ii]
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=[t0],
                                      end_frame_intervals=[t1], bins=bins,
                                      result_name=temp_name, category=self.category, replace=True)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(temp_name))
            plotter.plot(
                preplot=(fig, ax[j0, j1]), xlabel=r'$\theta$ (rad)', ylabel='PDF', ls='',
                label_suffix=' s', title=r't$\in$[%i, %i]s' % (t0/100, t1/100), c=cols[str(ii)],
                label='Experimental steady-state', normed=2)

            y = self.exp.get_df(temp_name).values.ravel()
            s = np.sum(y)
            y = y/s / dtheta/2.
            popt, _ = scopt.curve_fit(self._uniform_vonmises_dist, x, y, p0=0.2)
            q = round(popt[0], 3)
            y_fit = self._uniform_vonmises_dist(x, q)

            ax[j0, j1].plot(x, y_fit, c=cols[str(ii)], label='q='+str(q))
            ax[j0, j1].set_ylim(0, 0.5)
            plotter.draw_legend(ax[j0, j1])
        plotter.save(fig, name=result_name+'_evol_fit')

    @staticmethod
    def _uniform_vonmises_dist(x, q):
        kappa = 1.9
        y = q*scs.vonmises.pdf(x, kappa)+(1-q)/(2*np.pi)
        return y

    @staticmethod
    def _uniform_vonmises_dist2(x, q, kappa):
        y = q*scs.vonmises.pdf(x, kappa)+(1-q)/(2*np.pi)
        return y

    @staticmethod
    def _vonmises_dist(x, kappa):
        y = scs.vonmises.pdf(x, kappa)
        return y

    def compute_food_direction_error_hist_evol_around_first_attachment_fit(self):
        name = 'food_direction_error'
        first_attachment_name = 'first_attachment_time_of_outside_ant'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta/2., dtheta)

        dx = 0.25
        start_frame_intervals = np.arange(-.25, 2., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100

        self.exp.load(name)
        self.change_first_frame(name, first_attachment_name)
        self.exp.operation(name, np.abs)

        # self.exp.get_data_object(name).df = pd.concat([self.exp.get_df(name), -self.exp.get_df(name)])
        # self.exp.get_df(name).sort_index(inplace=True)

        result_name = self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                                end_frame_intervals=end_frame_intervals, bins=bins,
                                                category=self.category)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.create_plot()
        columns = self.exp.get_columns(result_name)
        cols = plotter.color_object.create_cmap('hot', columns)
        for column in columns:
            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name=column)
            plotter.plot(
                preplot=(fig, ax), xlabel=r'$\theta$ (rad)', ylabel='PDF', normed=True,
                c=cols[column], marker='.', ls='', label=r'$t\in$'+column+' s', display_legend=False)
            plotter.plot_fit(
                preplot=(fig, ax), typ='cst center gauss', c=cols[column], normed=True, display_legend=False)
        plotter.save(fig, name=result_name+'_fit')

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.create_plot(
            figsize=(8, 8), nrows=3, ncols=3, left=0.08, bottom=0.06, top=0.96, hspace=0.4, wspace=0.3)
        columns = self.exp.get_columns(result_name)
        cols = plotter.color_object.create_cmap('hot', columns)
        # n = int(5e5)
        for k, column in enumerate(columns):
            i = int(k/3)
            j = k-i*3

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name=column)
            plotter.plot(
                preplot=(fig, ax[i, j]), xlabel=r'$\theta$ (rad)', ylabel='PDF', normed=2,
                c=cols[column], marker='.', ls='', title=r'$t\in$'+column+' s', display_legend=False)
            b, a, c, _, _ = plotter.plot_fit(
                preplot=(fig, ax[i, j]), normed=2, typ='cst center gauss', c=cols[column], display_legend=False)

            # print(column, c, 1-np.pi*c)
            # if c < 1/np.pi:
            #     q = int(n*(1-np.pi*c))
            # else:
            #     q = 0
            # u = list(np.angle(np.exp(-1j*np.random.normal(scale=np.sqrt(0.8), size=q))))
            # v = list(np.pi-2*np.pi*np.random.uniform(size=n-q))
            # y, x = np.histogram(np.abs(u+v), np.arange(0, np.pi, 0.1), density=True)
            # ax[i, j].plot(x[:-1], y/2.)
            #
            ax[i, j].set_ylim(0, .5)

        plotter.save(fig, name=result_name+'_fit2')

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.create_plot(
            figsize=(8, 8), nrows=3, ncols=3, left=0.08, bottom=0.06, top=0.96, hspace=0.4, wspace=0.3)
        columns = self.exp.get_columns(result_name)
        cols = plotter.color_object.create_cmap('hot', columns)
        for k, column in enumerate(columns):
            i = int(k/3)
            j = k-i*3

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name=column)
            plotter.plot(
                preplot=(fig, ax[i, j]), xlabel=r'$\theta$ (rad)', ylabel='PDF', normed=2,
                c=cols[column], marker='.', ls='', title=r'$t\in$'+column+' s', display_legend=False)
            kappa, b, k, _, _ = plotter.plot_fit(
                preplot=(fig, ax[i, j]),  normed=2,
                typ='cst vonmises', c=cols[column], display_legend=False, p0=[1, 0.01, 0.1])
            print(column, kappa, b, k)

            ax[i, j].set_ylim(0, .5)

        plotter.save(fig, name=result_name+'_fit3')

    def compute_food_direction_error_hist_evol_around_first_attachment_fit_vonmises(self):
        name = 'food_direction_error'
        first_attachment_name = 'first_attachment_time_of_outside_ant'

        dtheta = np.pi/25.
        bins = np.arange(-np.pi+dtheta/2., np.pi+dtheta/2., dtheta)
        x = (bins[:-1] + bins[1:]) / 2.

        dx = 0.25
        start_frame_intervals = np.arange(-.25, 2., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100

        self.exp.load(name)
        self.change_first_frame(name, first_attachment_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(name))
        fig, ax = plotter.create_plot(
            figsize=(8, 8), nrows=3, ncols=3,  left=0.08, bottom=0.06, top=0.96, hspace=0.4, wspace=0.3)
        cols = plotter.color_object.create_cmap('hot', start_frame_intervals)

        # res = []
        for k, i in enumerate(range(len(start_frame_intervals))):
            j0 = int(k/3)
            j1 = k-j0*3
            ax[j0, j1].set_ylim(0, 0.5)

            frame0 = start_frame_intervals[i]
            frame1 = end_frame_intervals[i]

            vals = self.exp.get_df(name).loc[pd.IndexSlice[:, frame0:frame1], :].dropna().values.ravel()
            vals = np.array(list(vals)+list(-vals))

            if len(vals) != 0:
                # res.append((frame0/100., kappa))

                y, _ = np.histogram(vals, bins, density=True)
                ax[j0, j1].plot(x, y/2., 'o', c=cols[str(frame0)], mec='k')

                kappa, _, _ = scs.vonmises.fit(vals)
                ax[j0, j1].plot(x, scs.vonmises.pdf(x, kappa=kappa)/2., c=cols[str(frame0)])
                # _, scale = scs.norm.fit(vals, floc=0)
                # ax[j0, j1].plot(x, scs.norm.pdf(x, scale=scale), c=cols[str(frame0)])

        plotter.save(fig, name='%s_%s_%s' % (name, first_attachment_name, 'fit_vonmises'))

    def compute_food_direction_error_variance_evol_around_first_attachment(self, redo=False):
        name = 'food_direction_error'
        result_name = name + '_var_evol_around_first_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.2
        dx2 = 0.01
        start_frame_intervals = np.arange(-1, 3., dx2)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100

        label = 'Variance of the food direction error distribution over time'
        description = 'Variance of the angle between the food velocity and the food-exit vector,' \
                      'which gives in radian how much the food is not going in the good direction'

        if redo:

            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            self.exp.variance_evolution(name_to_var=name, start_index_intervals=start_frame_intervals,
                                        end_index_intervals=end_frame_intervals,
                                        category=self.category, result_name=result_name,
                                        label=label, description=description)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(
            xlabel='Time (s)', ylabel=r'$\sigma^2$ (rad$^2$)',
            label_suffix='s', label=r'$\sigma^2$', title='', marker='')
        plotter.plot_fit(typ='exp', preplot=(fig, ax), window=[90, 400], cst=(-0.01, .1, .1))
        plotter.draw_vertical_line(ax)
        plotter.save(fig)

    def compute_food_direction_error_variance_evol_around_first_attachment2(self, redo=False):
        name = 'food_direction_error'
        name2 = name + '_var_evol_around_first_attachment'
        result_name = name2 + '2'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.25
        dx2 = 0.01
        start_frame_intervals = np.arange(-.1, 3., dx2)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100

        dtheta = 0.1
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)

        label = 'Variance of the food direction error distribution over time'
        description = 'Variance of the angle between the food velocity and the food-exit vector,' \
                      'which gives in radian how much the food is not going in the good direction,' \
                      '(based on a gaussian fit)'

        if redo:

            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)
            # self.change_normalize_frame(name, init_frame_name)

            res = []
            for i in range(len(start_frame_intervals)):
                frame0 = start_frame_intervals[i]
                frame1 = end_frame_intervals[i]

                vals = self.exp.get_df(name).loc[pd.IndexSlice[:, frame0:frame1], :].values.ravel()

                if len(vals) != 0:

                    # y, _ = np.histogram(np.abs(vals), bins, density=True)
                    y, _ = np.histogram(list(vals)+list(-vals), bins, density=True)
                    x = (bins[:-1]+bins[1:])/2.

                    try:
                        c, s, d, _, _ = Fits.centered_gauss_cst_fit(x, y)
                        res.append((frame0/100., s**2, d, c))
                        print(frame0, frame1, res[-1])
                        # pb.plot(x, y, 'o')
                        # pb.plot(x_fit, y_fit)
                        # pb.show()
                    except RuntimeError:
                        pass

            res = np.array(res)
            # res[:, 0] = np.around(res[:, 0], 1)
            # res[:, 1:] = np.around(res[:, 1:], 3)
            self.exp.add_new_dataset_from_array(array=res, name=result_name, index_names='t',
                                                column_names=['variance', 'cst', 'cst2'],
                                                category=self.category, label=label,
                                                description=description)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name='cst')
        fig, ax = plotter.plot(
            xlabel='t (s)', ylabel='k(t)', label='k(t)', title='', marker='')
        a_c, b_c, c_c, _, _ = plotter.plot_fit(typ='exp', preplot=(fig, ax), window=[15, 250], cst=(-0.01, .1, .1))
        plotter.draw_vertical_line(ax)
        ax.set_ylim(0, .15)
        plotter.save(fig, result_name+'_cst')

        x = np.array(list(self.exp.get_index(result_name)))
        a_s, b_s, c_s, _, _ = self.exp.fit(
            name_to_fit=result_name, column='variance', typ='exp', window=[10, 100], cst=(-0.01, .1, .1))
        # a = Fits.exp_fct(x, a_s, b_s, c_s)
        c = Fits.exp_fct(x, a_c, b_c, c_c)
        self.exp.load(name2)
        a_var, b_var, c_var, _, _ = self.exp.fit(
            name_to_fit=name2, column='variance', typ='exp', window=[90, 400], cst=(-0.01, .1, .1))
        var = Fits.exp_fct(x, a_var, b_var, c_var)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name='cst2')
        fig, ax = plotter.plot(
            xlabel='t (s)', ylabel='d(t)', label='d(t)', title='', marker='')
        a_b, b_b, c_b, _, _ = plotter.plot_fit(typ='exp', preplot=(fig, ax), window=[15, 400], cst=(-.01, 0.5, .3))
        b = Fits.exp_fct(x, a_b, b_b, c_b)
        # ax.plot(x, (1 - 2 * np.pi * c) / (np.sqrt(2 * np.pi * a)))
        plotter.draw_vertical_line(ax)
        plotter.save(fig, result_name+'_cst2')

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name='variance')
        fig, ax = plotter.plot(
            xlabel='t (s)', ylabel=r'$s(t)^2$',
            label_suffix='s', label=r'$s(t)^2$', title='', marker='')
        ax.plot(x, 1/(2*np.pi)*((1-2*np.pi*c)/c)**2, label=r'$((1-2\pi\,k)/d)^2/(2\pi)$')
        ax.plot(x, (var-2/3.*c*np.pi**3)/(b*np.sqrt(np.pi)), label=r'$(\sigma^2(t)-2/3\pi^3 k)/(d\sqrt{\pi})$')

        plotter.plot_fit(typ='exp', preplot=(fig, ax), window=[10, 100], cst=(-0.01, .1, .1))
        plotter.draw_vertical_line(ax)
        ax.set_ylim(0, 1.2)
        plotter.save(fig)

    def compute_food_direction_error_variance_evol_around_first_attachment2_vonmises(self, redo=False):
        name = 'food_direction_error'
        name2 = name + '_var_evol_around_first_attachment'
        result_name = name2 + '2_vonmises'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 0.05
        dx2 = 2/6.
        start_frame_intervals = np.arange(-1, 4, dx) * 60 * 100
        end_frame_intervals = start_frame_intervals + dx2 * 60 * 100

        dtheta = 0.2
        bins = np.arange(-np.pi+dtheta/2., np.pi+dtheta/2, dtheta)

        label = 'Variance of the food direction error distribution over time'
        description = 'Variance of the angle between the food velocity and the food-exit vector,' \
                      'which gives in radian how much the food is not going in the good direction,' \
                      '(based on a von Mises fit)'

        if redo:

            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            res_q = []
            res_kappa = []
            for i in range(len(start_frame_intervals)):
                frame0 = start_frame_intervals[i]
                frame1 = end_frame_intervals[i]

                vals = self.exp.get_df(name).loc[pd.IndexSlice[:, frame0:frame1], :].values.ravel()

                if len(vals) != 0:

                    y, _ = np.histogram(list(vals)+list(-vals), bins, density=True)
                    x = (bins[:-1]+bins[1:])/2.

                    try:
                        # q = max(min(1-np.mean(y[x > 2.5])*2*np.pi, 1), 0)
                        # kappa, _, _ = Fits.uniform_vonmises_q_fit(x, y, q)
                        q, kappa, err, _, _ = Fits.uniform_vonmises_fit(x, y, get_err=True)
                        res_q.append((frame0/100., q, 1.96*err[0], 1.96*err[0]))
                        res_kappa.append((frame0/100., kappa, 1.96*err[1], 1.96*err[1]))

                        # pb.plot(x, y)
                        # pb.plot(x, Fits.uniform_vonmises_fct(x, q, kappa))
                        # pb.title((frame0/100, frame1/100, round(q, 2), round(kappa, 2)))
                        # pb.ylim(0, 0.5)
                        # pb.show()
                    except RuntimeError:
                        pass

            res_q = np.array(res_q)
            res_kappa = np.array(res_kappa)
            self.exp.add_new_dataset_from_array(array=np.around(res_q, 3), name=result_name+'_q', index_names='t',
                                                column_names=['q', 'err1', 'err2'],
                                                category=self.category, label=label, description=description)

            self.exp.add_new_dataset_from_array(array=np.around(res_kappa, 3), name=result_name+'_kappa',
                                                index_names='t', column_names=['kappa', 'err1', 'err2'],
                                                category=self.category, label=label, description=description)

            self.exp.write(result_name+'_q')
            self.exp.write(result_name+'_kappa')
        else:
            self.exp.load(result_name+'_q')
            self.exp.load(result_name+'_kappa')

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name+'_kappa'))
        fig, ax = plotter.plot_with_error(
            xlabel='t (s)', ylabel=r'$\kappa(t)$', label_suffix='s', label=r'$\kappa(t)$', title='', marker='')
        ax.set_xlim(-30, 120)
        ax.set_ylim(0, 5)
        plotter.draw_vertical_line(ax)
        plotter.save(fig)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name+'_q'))
        fig, ax = plotter.plot_with_error(xlabel='t (s)', ylabel='q(t)', label='q(t)', title='', marker='')
        # plotter.plot_fit(preplot=(fig, ax), typ='exp', cst=(-1, -1, 1))
        ax.set_xlim(-30, 120)
        ax.set_ylim(0, 1)
        plotter.draw_vertical_line(ax)
        plotter.save(fig, result_name+'_q')

        name_p_out = 'nb_outside_attachments_evol_around_first_outside_attachment'
        name_p_in = 'nb_inside_attachments_evol_around_first_outside_attachment'
        self.exp.load([name_p_out, name_p_in])

        self.exp.add_copy(name_p_out, 'temp', replace=True)
        self.exp.operation_between_2names('temp', name_p_in, lambda a, b: a/b, col_name1='p', col_name2='p')
        self.exp.operation_between_2names(
            'temp', result_name+'_q', lambda a, b: a*(1-b)/b, col_name1='p', col_name2='q')

        plotter2 = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'), column_name='p')
        fig, ax = plotter2.plot(
            xlabel='t (s)', ylabel=r'\alpha(t)', label=r'\alpha(t)', title='', marker='')
        ax.set_xlim(-30, 120)
        ax.set_ylim(0, 3)
        plotter.draw_vertical_line(ax)
        plotter.save(fig, result_name+'_q3')

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name+'_q'))
        fig, ax = plotter.plot(
            xlabel='t (s)', fct_y=lambda z: (1-z)/z, ylabel='(1-q)/q(t)', label='(1-q)/q(t)', title='', marker='')
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name+'_q'), column_name='q')
        plotter.plot_fit(preplot=(fig, ax), typ='exp', cst=(-1, -1, 1))
        ax.set_xlim(-30, 120)
        ax.set_ylim(0, 3)
        plotter.draw_vertical_line(ax)
        plotter.save(fig, result_name+'_q2')

        self.exp.remove_object(result_name+'_q')
        self.exp.remove_object(result_name+'_kappa')

    def compute_food_direction_error_variance_evol_around_first_attachment3(self, redo=False):
        name = 'food_direction_error'
        name2 = name + '_var_evol_around_first_attachment'
        result_name = name2 + '3'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 1
        dx2 = 0.01
        start_frame_intervals = np.arange(-.1, 3., dx2)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100

        dtheta = 0.1
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)

        label = 'Variance of the food direction error distribution over time'
        description = 'Variance of the angle between the food velocity and the food-exit vector,' \
                      'which gives in radian how much the food is not going in the good direction,' \
                      '(based on a gaussian fit)'

        if redo:

            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            res = []
            for i in range(len(start_frame_intervals)):
                frame0 = start_frame_intervals[i]
                frame1 = end_frame_intervals[i]

                vals = self.exp.get_df(name).loc[pd.IndexSlice[:, frame0:frame1], :].values.ravel()

                if len(vals) != 0:

                    y, _ = np.histogram(list(vals)+list(-vals), bins, density=True)
                    x = (bins[:-1]+bins[1:])/2.

                    try:
                        c, s, d, _, _ = Fits.centered_gauss_cst_fit(x, y)
                        res.append((frame0/100., s**2, d, c))
                        print(frame0, frame1, res[-1])
                    except RuntimeError:
                        pass

            res = np.array(res)
            self.exp.add_new_dataset_from_array(array=res, name=result_name, index_names='t',
                                                column_names=['variance', 'cst', 'cst2'],
                                                category=self.category, label=label,
                                                description=description)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name='variance')
        fig, ax = plotter.plot(
            xlabel='t (s)', ylabel=r'$s^2(t)$',
            label_suffix='s', label=r'$s^2(t)$', title='', marker='')
        # plotter.plot_fit(typ='cst', preplot=(fig, ax), window=[10, 100], cst=(-0.01, .1, .1))
        plotter.draw_vertical_line(ax)
        ax.set_ylim(0, 1.2)
        plotter.save(fig)

    def compute_food_direction_error_variance_evol_around_first_attachment3_vonmises(self, redo=False):
        name = 'food_direction_error'
        name2 = name + '_var_evol_around_first_attachment'
        result_name = name2 + '3_vonmises'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dx = 1
        dx2 = 0.01
        start_frame_intervals = np.arange(-.1, 3., dx2)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100

        dtheta = 0.1
        bins = np.arange(-np.pi-dtheta/2., np.pi+dtheta, dtheta)

        label = 'Variance of the food direction error distribution over time'
        description = 'Variance of the angle between the food velocity and the food-exit vector,' \
                      'which gives in radian how much the food is not going in the good direction,' \
                      '(based on a von Mises fit)'

        if redo:

            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            res = []
            for i in range(len(start_frame_intervals)):
                frame0 = start_frame_intervals[i]
                frame1 = end_frame_intervals[i]

                vals = self.exp.get_df(name).loc[pd.IndexSlice[:, frame0:frame1], :].values.ravel()

                if len(vals) != 0:

                    y, _ = np.histogram(list(vals)+list(-vals), bins, density=True)
                    x = (bins[:-1]+bins[1:])/2.

                    try:
                        kappa, d, k, _, _ = Fits.vonmises_cst_fit(x, y, p0=(1, 0.01, 0.1))
                        res.append((frame0/100., kappa, d, k))
                        print(res)
                    except RuntimeError:
                        pass

            res = np.array(res)
            self.exp.add_new_dataset_from_array(array=res, name=result_name, index_names='t',
                                                column_names=['kappa', 'd', 'k'],
                                                category=self.category, label=label,
                                                description=description)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name='kappa')
        fig, ax = plotter.plot(
            xlabel='t (s)', ylabel=r'$\kappa(t)$',
            label_suffix='kappa', label=r'$\kappa(t)$', title='', marker='')
        # plotter.plot_fit(typ='cst', preplot=(fig, ax), window=[10, 100], cst=(-0.01, .1, .1))
        plotter.draw_vertical_line(ax)
        ax.set_ylim(0, 5)
        plotter.save(fig)

    def compute_fisher_info_evol_around_first_attachment(self, redo=False):
        name = 'food_direction_error'
        result_name = 'fisher_info_evol_around_first_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        label = 'Fisher information over time'
        description = 'Fisher information over time'

        dx = 0.2
        dx2 = 0.01
        start_frame_intervals = np.arange(-1, 3., dx2)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100

        if redo:
            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            y = np.zeros((len(start_frame_intervals), 2))
            y[:, 0] = (end_frame_intervals + start_frame_intervals) / 2. / 100.
            for i in range(len(start_frame_intervals)):
                frame0 = start_frame_intervals[i]
                frame1 = end_frame_intervals[i]

                values = self.exp.get_df(name).loc[pd.IndexSlice[:, frame0:frame1], :].values.ravel()
                y[i, 1] = compute_fisher_information_uniform_von_mises(values)

            self.exp.add_new_dataset_from_array(array=y, name=result_name, index_names='t',
                                                column_names=['fisher_info'],
                                                category=self.category, label=label,
                                                description=description)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel=r'Fisher information',
                               label_suffix='s', label=r'Fisher information', marker='', title='', display_legend=False)
        # plotter.plot_fit(typ='exp', preplot=(fig, ax), window=[90, 400], cst=(-1, 1, 1))
        plotter.draw_vertical_line(ax)
        plotter.save(fig)

    def compute_mm10_food_direction_error(self, redo=False, redo_plot_indiv=False, redo_hist=False):

        name = 'food_direction_error'
        result_name = 'mm10_' + name
        food_exit_angle_name = 'mm10_food_exit_angle'
        vel_name = 'mm10_food_velocity'
        time_window = 10
        self.__compute_food_direction_error(result_name, self.category, food_exit_angle_name, vel_name,
                                            time_window, redo, redo_hist, redo_plot_indiv)

    def compute_mm20_food_direction_error(self, redo=False, redo_plot_indiv=False, redo_hist=False):

        name = 'food_direction_error'
        result_name = 'mm20_' + name
        food_exit_angle_name = 'mm10_food_exit_angle'
        vel_name = 'mm20_food_velocity'
        time_window = 20
        self.__compute_food_direction_error(result_name, self.category, food_exit_angle_name, vel_name,
                                            time_window, redo, redo_hist, redo_plot_indiv)

    def compute_mm1s_food_direction_error(self, redo=False, redo_plot_indiv=False, redo_hist=False):

        name = 'food_direction_error'
        result_name = 'mm1s_' + name
        food_exit_angle_name = 'mm1s_food_exit_angle'
        vel_name = 'mm1s_food_velocity'
        time_window = 100
        self.__compute_food_direction_error(result_name, self.category, food_exit_angle_name, vel_name,
                                            time_window, redo, redo_hist, redo_plot_indiv)

    def compute_mm10s_food_direction_error(self, redo=False, redo_plot_indiv=False, redo_hist=False):

        name = 'food_direction_error'
        result_name = 'mm10s_' + name
        food_exit_angle_name = 'mm10s_food_exit_angle'
        vel_name = 'mm10s_food_velocity'
        time_window = 1000
        self.__compute_food_direction_error(result_name, self.category, food_exit_angle_name, vel_name,
                                            time_window, redo, redo_hist, redo_plot_indiv)

    def compute_mm30s_food_direction_error(self, redo=False, redo_plot_indiv=False, redo_hist=False):

        name = 'food_direction_error'
        result_name = 'mm30s_' + name
        food_exit_angle_name = 'mm30s_food_exit_angle'
        vel_name = 'mm30s_food_velocity'
        time_window = 3000
        self.__compute_food_direction_error(result_name, self.category, food_exit_angle_name, vel_name,
                                            time_window, redo, redo_hist, redo_plot_indiv)

    def __compute_food_direction_error(self, result_name, category, food_exit_angle_name, vel_name, time_window,
                                       redo, redo_hist, redo_plot_indiv):
        dtheta = np.pi / 12.
        bins = np.around(np.arange(-np.pi + dtheta / 2., np.pi + dtheta, dtheta), 3)
        result_label = 'Food direction error'
        result_description = 'Angle between the food-exit angle and the angular coordinate of the food trajectory, ' \
                             'both angles are smoothed by a moving mean of window '+str(time_window)+' frames'

        if redo:
            vel_name_x = vel_name+'_x'
            vel_name_y = vel_name+'_y'
            self.exp.load([food_exit_angle_name, vel_name_x, vel_name_y])

            vel = pd.DataFrame(index=self.exp.get_index(vel_name_x))
            vel['x'] = self.exp.get_df(vel_name_x)
            vel['y'] = self.exp.get_df(vel_name_y)

            vel_phi = Geo.angle_df(vel)

            tab = Geo.angle_distance(self.exp.get_df(food_exit_angle_name)[food_exit_angle_name], vel_phi)

            self.exp.add_copy1d(name_to_copy=vel_name_x, copy_name=result_name, category=category,
                                label=result_label, description=result_description)

            self.exp.change_values(result_name, np.around(tab, 6))

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        self.__plot_indiv(result_name, redo, redo_plot_indiv)

        hist_name = self.compute_hist(name=result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter2 = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig2, ax2 = plotter2.plot(xlabel='Food direction error', ylabel='PDF', normed=True)
        plotter2.save(fig2)

    def __plot_indiv(self, result_name, redo, redo_plot_indiv):
        if redo or redo_plot_indiv:
            attachment_name = 'outside_ant_carrying_intervals'
            self.exp.load(['fps', attachment_name])

            def plot4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                fps = self.exp.get_value('fps', id_exp)
                df2 = df.loc[id_exp, :]
                df2 = df2.abs()
                df2.index = df2.index / fps

                self.exp.add_new_dataset_from_df(df=df2, name='temp', category=self.category, replace=True)

                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
                fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Food direction error', marker='', title=id_exp)

                attachments = self.exp.get_df(attachment_name).loc[id_exp, :]
                attachments.reset_index(inplace=True)
                attachments = np.array(attachments)

                colors = plotter.color_object.create_cmap('hot_r', set(list(attachments[:, 0])))
                for id_ant, frame, inter in attachments:
                    ax.axvline(frame / fps, c=colors[str(id_ant)], alpha=0.5)

                ax.set_ylim((0, np.pi))
                plotter.save(fig, name=id_exp, sub_folder=result_name)

            self.exp.groupby(result_name, id_exp_name, plot4each_group)

    def veracity_over_derivative(self):
        name = 'mm10_food_direction_error'
        res_name = 'food_direction_error_over_difference'
        self.exp.load([name, 'fps'])
        label = 'Food direction error difference as a function of food direction error'

        dt = 0.1

        def get_diff4each_group(df: pd.DataFrame):
            id_exp = df.index.get_level_values(id_exp_name)[0]
            fps = self.exp.get_value('fps', id_exp)

            dframe = int(dt*fps/2)

            df2 = df.loc[id_exp, :].copy()
            df3 = df.loc[id_exp, :].copy()

            df2.index += dframe
            df3.index -= dframe

            df2 = df2.reindex(df.index.get_level_values(id_frame_name))
            df3 = df3.reindex(df.index.get_level_values(id_frame_name))

            df.loc[id_exp, :] = Geo.angle_distance(df3, df2).ravel()

            return df

        df_diff = self.exp.groupby(name, id_exp_name, get_diff4each_group)
        self.exp.add_new1d_from_df(df=np.abs(df_diff), name=name+'_diff', object_type='CharacteristicTimeSeries1d')
        # idx = pd.Index(self.exp.get_df(name).values.ravel(), name='food_direction_error')
        # df_res = pd.DataFrame(df_diff.values, index=idx, columns=['difference'])
        #
        # self.exp.add_new_dataset_from_df(df=df_res, name=res_name, category=self.category,
        #                                  label=label, description=label)

        self.exp.operation(name, np.abs)

        self.exp.hist2d(xname_to_hist=name, yname_to_hist=name+'_diff', result_name=res_name,
                        category=self.category, bins=np.arange(0, np.pi, np.pi/50.),
                        label=label, description=label)

        # self.exp.vs(xname=name, yname=name+'_diff', result_name=res_name,
        #             category=self.category, n_bins=50,
        #             label=label, description=label)
        self.exp.write(res_name)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(res_name))
        fig, ax = plotter.create_plot(figsize=(10, 8))
        plotter.plot_heatmap(preplot=(fig, ax), cmap_scale_log=True)
        # fig, ax = plotter.plot_with_error()
        plotter.save(fig)

    def compute_veracity_derivative(self, redo, redo_hist=False):
        res_name = 'food_direction_error_derivative'
        label = 'Food direction error derivative'
        description = 'Food direction error derivative'

        dtheta = np.pi/100.
        bins = np.arange(-np.pi+dtheta/2., np.pi+dtheta/2., dtheta)

        if redo:
            name = 'mm10_food_direction_error'
            self.exp.load([name, 'fps'])
            dt = 0.1

            def get_diff4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                fps = self.exp.get_value('fps', id_exp)

                dframe = int(dt*fps/2)

                df2 = df.loc[id_exp, :].copy()
                df3 = df.loc[id_exp, :].copy()

                df2.index += dframe
                df3.index -= dframe

                df2 = df2.reindex(df.index.get_level_values(id_frame_name))
                df3 = df3.reindex(df.index.get_level_values(id_frame_name))

                df.loc[id_exp, :] = Geo.angle_distance(df3, df2).ravel()

                return df

            df_diff = self.exp.groupby(name, id_exp_name, get_diff4each_group)
            self.exp.add_new1d_from_df(
                df=df_diff, name=res_name, object_type='CharacteristicTimeSeries1d',
                category=self.category, label=label, description=description)

            self.exp.write(res_name)

        hist_name = self.compute_hist(name=res_name, bins=bins, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(self.exp.root, obj=self.exp.get_data_object(hist_name))

        fig, ax = plotter.plot(normed=True, display_legend=False)
        plotter.plot_fit(preplot=(fig, ax), typ='cst center gauss', normed=True, display_legend=False)

        n = int(1e6)
        q = int(n*0.5)
        p = int(n*0.4)
        u = list(np.angle(np.exp(-1j*scs.norm.rvs(scale=0.3, size=q))))
        v = list(np.pi*(1-2*np.random.uniform(size=n-q-p)))
        w = list(np.random.laplace(scale=0.5, loc=0, size=p))
        y, x = np.histogram(u+v+w, np.arange(-np.pi-0.005, np.pi, 0.01), density=True)
        ax.plot(x[:-1]+0.005, y)
        plotter.save(fig)

    @staticmethod
    def temp():
        t = np.array([0, 10, 60, 80])
        a = np.exp(-0.062)
        b = 3.066
        c = 0.645
        y = Fits.exp_fct(t, a, b, c)

        def equations(p):
            a_out, b_out, a_in, b_in = p
            res = b_out*np.exp(-a_out*t)+0.05-(b_in*np.exp(-a_in*t)+0.03)*y
            return res

        p0 = np.array([0.9, -0.04, 0.9, 0.074])
        print(scopt.fsolve(equations, p0))

    def compute_mm1s_food_direction_error_variation(self, redo, redo_hist=False):
        dtheta = .25
        bins = np.arange(0, np.pi + dtheta / 2., dtheta)
        # bins = np.array(list(-bins[::-1]) + list(bins))

        name_error = 'mm1s_food_direction_error'
        result_name = name_error+'_variation'
        label = 'Food directional error variation'
        description = 'Food directional error difference between 0s and 2s'
        self._get_directional_error_variation(result_name, name_error, label, description, redo)

        hist_name = self.compute_hist(result_name, bins=bins, redo=redo, redo_hist=redo_hist)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(xlabel='(rad)', ylabel='PDF', title='', normed=True)
        ax.set_ylim(0, 1.)
        plotter.save(fig)

        surv_name = self.compute_surv(result_name, redo=redo, redo_hist=redo_hist)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(surv_name))
        fig, ax = plotter.plot(xlabel='(rad)', ylabel='PDF', title='', marker=None)
        ax.set_ylim(0, 1.)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(.05))
        ax.grid(which='both')
        plotter.save(fig)

    def _get_directional_error_variation(self, result_name, name_error, label, description, redo):
        if redo:
            self.exp.load(name_error)

            self.exp.add_copy(old_name=name_error, new_name=result_name, category=self.category,
                              label=label, description=description)
            index = self.exp.get_index(name_error)

            df_error0 = self.exp.get_df(name_error).copy()
            df_error0.reset_index(inplace=True)
            df_error0[id_frame_name] += 100
            df_error0.set_index([id_exp_name, id_frame_name], inplace=True)

            df_error1 = self.exp.get_df(name_error).copy()
            df_error1.reset_index(inplace=True)
            df_error1[id_frame_name] -= 100
            df_error1.set_index([id_exp_name, id_frame_name], inplace=True)

            df_error0 = df_error0.reindex(index)
            df_error1 = df_error1.reindex(index)
            df_error_diff0 = Geo.angle_distance_df(df_error0, df_error1)
            self.exp.get_df(result_name)[:] = np.abs(np.around(np.c_[df_error_diff0[:]], 3))

            self.exp.write(result_name)

    def compute_food_direction_error_variation_hist_evol_around_first_outside_attachment(self, redo=False):
        name = 'mm1s_food_direction_error_variation'
        result_name = name+'_hist_evol_around_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        dx = 0.25
        start_frame_intervals = np.array(np.arange(0, 3.5, dx)*60*100, dtype=int)
        end_frame_intervals = np.array(start_frame_intervals + dx*60*100*2, dtype=int)

        hist_label = 'Food direction error variation distribution over time (rad)'
        hist_description = 'Histogram over time of the food direction error variation (rad)'

        if redo:
            self.exp.load(name)
            self.change_first_frame(name, init_frame_name)

            self.exp.operation(name, np.abs)
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=start_frame_intervals,
                                      end_frame_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category,
                                      label=hist_label, description=hist_description)
            self.exp.remove_object(name)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta_{error}$ (rad)', ylabel='PDF', normed=True, label_suffix='s',
                               title='')
        # ax.set_ylim((0, 1))
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def compute_food_direction_error_hist_evol_w10s_path_efficiency(self, redo=False):
        w = 10
        discrim_name = 'w'+str(w)+'s_food_path_efficiency'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        d_eff = 0.1
        start_eff_intervals = np.around(np.arange(0, 1, d_eff), 1)
        end_eff_intervals = np.around(start_eff_intervals+d_eff, 1)

        result_name = self._get_food_direction_error_over_path_efficiency(
            bins, discrim_name, end_eff_intervals, redo, start_eff_intervals, w)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta_{error}$ (rad)', ylabel='PDF', normed=2,
                               title='')
        ax.set_ylim((0, 0.6))
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def compute_food_direction_error_hist_evol_w16s_path_efficiency(self, redo=False):
        w = 16
        discrim_name = 'w'+str(w)+'s_food_path_efficiency'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        d_eff = 0.1
        start_eff_intervals = np.around(np.arange(0, 1, d_eff), 1)
        end_eff_intervals = np.around(start_eff_intervals+d_eff, 1)

        result_name = self._get_food_direction_error_over_path_efficiency(
            bins, discrim_name, end_eff_intervals, redo, start_eff_intervals, w)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta_{error}$ (rad)', ylabel='PDF', normed=2,
                               title='')
        ax.set_ylim((0, 0.6))
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def _get_food_direction_error_over_path_efficiency(self, bins, discrim_name, end_eff_intervals, redo,
                                                       start_eff_intervals, w):
        name = 'food_direction_error'
        temp_name = '%s_%s' % (name, discrim_name)
        result_name = '%s_hist_evol_%s' % (name, discrim_name)
        last_frame_name = 'food_exit_frames'
        label = 'Food direction error distribution over path efficiency (rad)'
        description = 'Histogram of the angle between the food velocity and the food-exit vector,' \
                      'which gives in radian how much the food is not going in the good direction (rad)'
        if redo:
            self.exp.load([name, discrim_name, last_frame_name])
            self.cut_last_frames_for_indexed_by_exp_frame_indexed(name, last_frame_name)

            self._add_path_efficiency_index(name, discrim_name, temp_name, w=w)

            self.exp.operation(temp_name, np.abs)
            self.exp.hist1d_evolution(name_to_hist=temp_name, start_frame_intervals=start_eff_intervals,
                                      end_frame_intervals=end_eff_intervals, bins=bins, index_name=discrim_name,
                                      result_name=result_name, category=self.category,
                                      label=label, description=description)

            self.exp.remove_object(name)
            self.exp.remove_object(discrim_name)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        return result_name

    def _add_path_efficiency_index(self, name, discrim_name, temp_name, w=10):
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

    def compute_food_direction_error_hist_evol_w10s_path_efficiency_fit(self):
        w = 10
        discrim_name = 'w'+str(w)+'s_food_path_efficiency'

        start_eff_intervals = np.around([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 2)
        end_eff_intervals = np.around([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 2)

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        self._get_food_direction_error_over_path_efficiency_fit(
            bins, discrim_name, dtheta, end_eff_intervals, start_eff_intervals)

    def compute_food_direction_error_hist_evol_w16s_path_efficiency_fit(self):
        w = 16
        discrim_name = 'w'+str(w)+'s_food_path_efficiency'

        start_eff_intervals = np.around([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 2)
        end_eff_intervals = np.around([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 2)

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        self._get_food_direction_error_over_path_efficiency_fit(
            bins, discrim_name, dtheta, end_eff_intervals, start_eff_intervals)

    def _get_food_direction_error_over_path_efficiency_fit(self, bins, discrim_name, dtheta, end_eff_intervals,
                                                           start_eff_intervals):
        variable_name = 'food_direction_error'
        last_frame_name = 'food_exit_frames'
        name = '%s_hist_evol_%s' % (variable_name, discrim_name)
        x = (bins[1:] + bins[:-1]) / 2.
        self.exp.load([variable_name, discrim_name, last_frame_name])
        self.cut_last_frames_for_indexed_by_exp_frame_indexed(variable_name, last_frame_name)
        self._add_path_efficiency_index(variable_name, discrim_name, name)
        self.exp.operation(name, np.abs)
        temp_name = 'temp'
        plotter = BasePlotters()
        cols = plotter.color_object.create_cmap('hot', range(len(start_eff_intervals)))
        fig, ax = plotter.create_plot(
            figsize=(8, 8), nrows=3, ncols=3, left=0.08, bottom=0.06, top=0.96, hspace=0.4, wspace=0.3)
        for ii in range(len(start_eff_intervals)):
            j0 = int(ii / 3)
            j1 = ii - j0 * 3
            eff0 = start_eff_intervals[ii]
            eff1 = end_eff_intervals[ii]
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=[eff0],
                                      end_frame_intervals=[eff1], bins=bins, index_name=discrim_name,
                                      result_name=temp_name, category=self.category, replace=True)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(temp_name))
            plotter.plot(
                preplot=(fig, ax[j0, j1]), xlabel=r'$\theta$ (rad)', ylabel='PDF', ls='',
                label_suffix=' s', title=r'Eff.$\in$[%.2f, %.2f]' % (eff0, eff1), c=cols[str(ii)],
                label='Experiment', normed=2)

            y = self.exp.get_df(temp_name).values.ravel()
            s = np.sum(y)
            y = y / s / dtheta / 2.
            popt, _ = scopt.curve_fit(self._uniform_vonmises_dist2, x, y, p0=[0.2, 2.], bounds=(0, [1, np.inf]))
            q = round(popt[0], 3)
            kappa = round(popt[1], 3)
            # kappa = 1.9

            y_fit = self._uniform_vonmises_dist2(x, q, kappa)

            ax[j0, j1].plot(x, y_fit, c=cols[str(ii)], label=r'q=%.3f, $\kappa$=%.3f' % (q, kappa))
            ax[j0, j1].set_ylim(0, 1)
            plotter.draw_legend(ax[j0, j1])
        plotter.save(fig, name=name + '_fit')

    def compute_food_direction_error_hist_evol_w10s_path_efficiency_resolution1pc(self, redo=False):
        w = 10
        self._get_direction_error_over_path_efficiency_resolution1pc(w, redo)

    def compute_food_direction_error_hist_evol_w16s_path_efficiency_resolution1pc(self, redo=False):
        w = 16
        self._get_direction_error_over_path_efficiency_resolution1pc(w, redo)

    def _get_direction_error_over_path_efficiency_resolution1pc(self, w, redo):
        discrim_name = 'w' + str(w) + 's_food_path_efficiency_resolution1pc'
        name = 'food_direction_error'
        last_frame_name = 'food_exit_frames'

        temp_name = '%s_%s' % (name, discrim_name)
        result_name = '%s_hist_evol_%s' % (name, discrim_name)

        dtheta = np.pi / 12.
        bins = np.arange(0, np.pi + dtheta, dtheta)

        start_eff_intervals = [0, 0.3, 0.5, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975]
        end_eff_intervals = [0.3, 0.5, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1]

        label = 'Food direction error distribution over path efficiency (rad)'
        description = 'Histogram of the angle between the food velocity and the food-exit vector,' \
                      'which gives in radian how much the food is not going in the good direction (rad)'
        if redo:
            self.exp.load([name, discrim_name, last_frame_name])
            self.exp.change_df(name, self.exp.get_df(name).reindex(self.exp.get_index(discrim_name)))

            self.cut_last_frames_for_indexed_by_exp_frame_indexed(name, last_frame_name)

            self._add_path_efficiency_index(name, discrim_name, temp_name, w=w)

            self.exp.operation(temp_name, np.abs)
            self.exp.hist1d_evolution(name_to_hist=temp_name, start_frame_intervals=start_eff_intervals,
                                      end_frame_intervals=end_eff_intervals, bins=bins, index_name=discrim_name,
                                      result_name=result_name, category=self.category,
                                      label=label, description=description)

            self.exp.remove_object(name)
            self.exp.remove_object(discrim_name)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta_{error}$ (rad)', ylabel='PDF', normed=2,
                               title='')
        ax.set_ylim((0, 0.6))
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def compute_food_direction_error_hist_evol_w10s_path_efficiency_resolution1pc_fit(self):
        w = 10
        self._get_direction_error_over_path_efficiency_resolution1pc_fit(w)

    def compute_food_direction_error_hist_evol_w16s_path_efficiency_resolution1pc_fit(self):
        w = 16
        self._get_direction_error_over_path_efficiency_resolution1pc_fit(w)

    def _get_direction_error_over_path_efficiency_resolution1pc_fit(self, w):
        discrim_name = 'w' + str(w) + 's_food_path_efficiency_resolution1pc'
        variable_name = 'food_direction_error'
        last_frame_name = 'food_exit_frames'

        name = '%s_hist_evol_%s' % (variable_name, discrim_name)

        start_eff_intervals = np.around([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 2)
        end_eff_intervals = np.around([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 2)

        dtheta = np.pi / 25.
        bins = np.arange(0, np.pi + dtheta, dtheta)
        x = (bins[1:] + bins[:-1]) / 2.

        self.exp.load([variable_name, discrim_name, last_frame_name])
        self.exp.change_df(variable_name, self.exp.get_df(variable_name).reindex(self.exp.get_index(discrim_name)))
        self.cut_last_frames_for_indexed_by_exp_frame_indexed(variable_name, last_frame_name)

        self._add_path_efficiency_index(variable_name, discrim_name, name)
        self.exp.operation(name, np.abs)

        temp_name = 'temp'
        plotter = BasePlotters()
        cols = plotter.color_object.create_cmap('hot', range(len(start_eff_intervals)))
        fig, ax = plotter.create_plot(
            figsize=(8, 8), nrows=3, ncols=3, left=0.08, bottom=0.06, top=0.96, hspace=0.4, wspace=0.3)
        for ii in range(len(start_eff_intervals)):
            j0 = int(ii / 3)
            j1 = ii - j0 * 3
            eff0 = start_eff_intervals[ii]
            eff1 = end_eff_intervals[ii]
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=[eff0],
                                      end_frame_intervals=[eff1], bins=bins, index_name=discrim_name,
                                      result_name=temp_name, category=self.category, replace=True)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(temp_name))
            plotter.plot(
                preplot=(fig, ax[j0, j1]), xlabel=r'$\theta$ (rad)', ylabel='PDF', ls='',
                label_suffix=' s', title=r'Eff.$\in$[%.2f, %.2f]' % (eff0, eff1), c=cols[str(ii)],
                label='Experiment', normed=2)

            y = self.exp.get_df(temp_name).values.ravel()
            s = np.sum(y)
            y = y / s / dtheta / 2.
            popt, _ = scopt.curve_fit(self._uniform_vonmises_dist2, x, y, p0=[0.2, 2.], bounds=(0, [1, np.inf]))
            q = round(popt[0], 3)
            kappa = round(popt[1], 3)
            # kappa = 1.9

            y_fit = self._uniform_vonmises_dist2(x, q, kappa)

            ax[j0, j1].plot(x, y_fit, c=cols[str(ii)], label=r'q=%.3f, $\kappa$=%.3f' % (q, kappa))
            ax[j0, j1].set_ylim(0, 1)
            plotter.draw_legend(ax[j0, j1])
        plotter.save(fig, name=name + '_fit')

    def compute_food_direction_error_hist_evol_attachment_rate(self, redo=False):
        w = 10
        discrim_name = 'attachments'
        discrim_name2 = 'attachment_rate'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        start_rate_intervals = np.around([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 1)
        end_rate_intervals = np.around([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2], 1)

        result_name = self._get_food_direction_error_over_attachment_rate(
            bins, discrim_name, discrim_name2, end_rate_intervals, redo, start_rate_intervals, w)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta_{error}$ (rad)', ylabel='PDF', normed=2,
                               title='Directional error over attachment rate', label_suffix=r's$^{-1}$')
        ax.set_ylim((0, 0.6))
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def compute_food_direction_error_hist_evol_outside_attachment_rate(self, redo=False):
        w = 10
        discrim_name = 'outside_attachments'
        discrim_name2 = 'outside_attachment_rate'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        start_rate_intervals = np.around([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 1)
        end_rate_intervals = np.around([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2], 1)

        result_name = self._get_food_direction_error_over_attachment_rate(
            bins, discrim_name, discrim_name2, end_rate_intervals, redo, start_rate_intervals, w)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta_{error}$ (rad)', ylabel='PDF', normed=2,
                               title='Directional error over outside attachment rate', label_suffix=r's$^{-1}$')
        ax.set_ylim((0, 0.6))
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def compute_food_direction_error_hist_evol_inside_attachment_rate(self, redo=False):
        w = 10
        discrim_name = 'inside_attachments'
        discrim_name2 = 'inside_attachment_rate'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        start_rate_intervals = np.around([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 1)
        end_rate_intervals = np.around([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2], 1)

        result_name = self._get_food_direction_error_over_attachment_rate(
            bins, discrim_name, discrim_name2, end_rate_intervals, redo, start_rate_intervals, w)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta_{error}$ (rad)', ylabel='PDF', normed=2,
                               title='Directional error over inside attachment rate', label_suffix=r's$^{-1}$')
        ax.set_ylim((0, 0.6))
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def compute_food_direction_error_hist_evol_attachment_prop(self, redo=False):
        w = 10
        discrim_name = 'attachments'
        discrim_name2 = 'attachment_prop_rate'

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        d_rate = 0.1
        start_rate_intervals = np.around(np.arange(0, 1, d_rate), 1)
        end_rate_intervals = np.around(start_rate_intervals+d_rate, 1)

        result_name = self._get_food_direction_error_over_attachment_prop_rate(
            bins, discrim_name, discrim_name2, end_rate_intervals, redo, start_rate_intervals, w)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta_{error}$ (rad)', ylabel='PDF', normed=2,
                               title='Directional error over prop. of outside attachments')
        ax.set_ylim((0, 0.6))
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def _get_food_direction_error_over_attachment_rate(self, bins, discrim_name, discrim_name2, end_eff_intervals, redo,
                                                       start_eff_intervals, w=10):
        name = 'food_direction_error'
        temp_name = '%s_%s' % (name, discrim_name)
        last_frame_name = 'food_exit_frames'

        result_name = '%s_hist_evol_%s' % (name, discrim_name2)
        label = 'Food direction error distribution over attachment rate (rad)'
        description = 'Histogram of the angle between the food velocity and the food-exit vector,' \
                      'which gives in radian how much the food is not going in the good direction (rad)'
        if redo:
            self.exp.load([name, discrim_name, last_frame_name])
            self.cut_last_frames_for_indexed_by_exp_frame_indexed(name, last_frame_name)

            self._get_rolling_rate(discrim_name, w)
            self._add_attachment_rate_index(name, discrim_name, temp_name)

            self.exp.operation(temp_name, np.abs)
            self.exp.hist1d_evolution(name_to_hist=temp_name, start_frame_intervals=start_eff_intervals,
                                      end_frame_intervals=end_eff_intervals, bins=bins, index_name=discrim_name,
                                      result_name=result_name, category=self.category,
                                      label=label, description=description)

            self.exp.remove_object(name)
            self.exp.remove_object(discrim_name)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        return result_name

    def _add_attachment_rate_index(self, name, discrim_name, temp_name):

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

    def _get_rolling_prop(self, discrim_name, w):
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

    def _get_rolling_rate(self, discrim_name, w):
        df = self.exp.get_df(discrim_name).copy()
        df.reset_index(inplace=True)
        df.drop(columns=id_ant_name, inplace=True)
        df.set_index([id_exp_name, id_frame_name], inplace=True)
        df = df.rolling(window=w * 100, min_periods=100).sum() / 10.
        self.exp.change_df(discrim_name, df)

    def _get_food_direction_error_over_attachment_prop_rate(
            self, bins, discrim_name, discrim_name2, end_eff_intervals, redo, start_eff_intervals, w):

        name = 'food_direction_error'
        last_frame_name = 'food_exit_frames'

        temp_name = '%s_%s' % (name, discrim_name)
        result_name = '%s_hist_evol_%s' % (name, discrim_name2)
        label = 'Food direction error distribution over attachment rate (rad)'
        description = 'Histogram of the angle between the food velocity and the food-exit vector,' \
                      'which gives in radian how much the food is not going in the good direction (rad)'
        if redo:
            self.exp.load([name, 'outside_'+discrim_name, 'inside_'+discrim_name, last_frame_name])
            self.cut_last_frames_for_indexed_by_exp_frame_indexed(name, last_frame_name)

            self._get_rolling_prop(discrim_name, w)
            self._add_attachment_rate_index(name, 'outside_'+discrim_name, temp_name)

            self.exp.operation(temp_name, np.abs)
            self.exp.hist1d_evolution(name_to_hist=temp_name, start_frame_intervals=start_eff_intervals,
                                      end_frame_intervals=end_eff_intervals, bins=bins,
                                      index_name='outside_'+discrim_name,
                                      result_name=result_name, category=self.category,
                                      label=label, description=description)

            self.exp.remove_object(name)
            self.exp.remove_object('outside_'+discrim_name)
            self.exp.remove_object('inside_'+discrim_name)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        return result_name

    def compute_food_direction_error_hist_evol_outside_attachment_rate_fit(self):
        discrim_name = 'outside_attachments'
        variable_name = 'food_direction_error'
        last_frame_name = 'food_exit_frames'
        result_name = '%s_hist_evol_%s' % (variable_name, discrim_name)

        start_eff_intervals = np.around([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 1)
        end_eff_intervals = np.around([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2], 1)

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        self.exp.load([variable_name, discrim_name, last_frame_name])
        self.cut_last_frames_for_indexed_by_exp_frame_indexed(variable_name, last_frame_name)

        self._get_rolling_rate(discrim_name, 10)
        self._add_attachment_rate_index(variable_name, discrim_name, result_name)

        self._get_food_direction_error_over_attachment_rate_fit(
            result_name, bins, discrim_name, dtheta, end_eff_intervals, start_eff_intervals)

    def compute_food_direction_error_hist_evol_inside_attachment_rate_fit(self):
        discrim_name = 'inside_attachments'

        start_eff_intervals = np.around([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 1)
        end_eff_intervals = np.around([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 2], 1)

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        variable_name = 'food_direction_error'
        result_name = '%s_hist_evol_%s' % (variable_name, discrim_name)
        self.exp.load([variable_name, discrim_name])
        self._get_rolling_rate(discrim_name, 10)
        self._add_attachment_rate_index(variable_name, discrim_name, result_name)

        self._get_food_direction_error_over_attachment_rate_fit(
            result_name, bins, discrim_name, dtheta, end_eff_intervals, start_eff_intervals)

    def compute_food_direction_error_hist_evol_attachment_prop_fit(self):
        discrim_name = 'attachments'
        discrim_name2 = 'attachment_prop_rate'
        variable_name = 'food_direction_error'
        last_frame_name = 'food_exit_frames'
        result_name = '%s_hist_evol_%s' % (variable_name, discrim_name2)

        start_eff_intervals = np.around([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 1)
        end_eff_intervals = np.around([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 1)

        dtheta = np.pi/25.
        bins = np.arange(0, np.pi+dtheta, dtheta)

        self.exp.load([variable_name, last_frame_name])
        self.cut_last_frames_for_indexed_by_exp_frame_indexed(variable_name, last_frame_name)

        self._get_rolling_prop(discrim_name, 10)
        self._add_attachment_rate_index(variable_name, 'outside_'+discrim_name, result_name)

        self._get_food_direction_error_over_attachment_rate_fit(
            result_name, bins, 'outside_'+discrim_name, dtheta, end_eff_intervals, start_eff_intervals,
            title=r'Prop. $\in$[%.1f, %.1f]')

    def _get_food_direction_error_over_attachment_rate_fit(self, name, bins, discrim_name, dtheta,
                                                           end_eff_intervals, start_eff_intervals,
                                                           title=r'Rate $\in$[%.1f, %.1f] $s^{-1}$'):

        x = (bins[1:] + bins[:-1]) / 2.
        self.exp.operation(name, np.abs)
        temp_name = 'temp'
        plotter = BasePlotters()
        cols = plotter.color_object.create_cmap('hot', range(len(start_eff_intervals)))
        fig, ax = plotter.create_plot(
            figsize=(8, 8), nrows=3, ncols=3, left=0.08, bottom=0.06, top=0.96, hspace=0.4, wspace=0.3)
        for ii in range(len(start_eff_intervals)):
            j0 = int(ii / 3)
            j1 = ii - j0 * 3
            eff0 = start_eff_intervals[ii]
            eff1 = end_eff_intervals[ii]
            self.exp.hist1d_evolution(name_to_hist=name, start_frame_intervals=[eff0],
                                      end_frame_intervals=[eff1], bins=bins, index_name=discrim_name,
                                      result_name=temp_name, category=self.category, replace=True)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(temp_name))
            plotter.plot(
                preplot=(fig, ax[j0, j1]), xlabel=r'$\theta$ (rad)', ylabel='PDF', ls='',
                label_suffix=' s', title=title % (eff0, eff1), c=cols[str(ii)],
                label='Experiment', normed=2)

            y = self.exp.get_df(temp_name).values.ravel()
            s = np.nansum(y)
            y = y / s / dtheta / 2.

            mask = np.where(x > 2.)
            y2 = y[mask]
            x2 = x[mask]
            popt, _ = scopt.curve_fit(lambda z, c: (1-c)/(2*np.pi), x2, y2, p0=0.5, bounds=(0, 1))
            q = popt[0]

            mask = np.where(~np.isnan(y))
            y = y[mask]
            x2 = x[mask]
            popt, _ = scopt.curve_fit(
                lambda z, k: q*scs.vonmises.pdf(x, k)+(1-q)/(2*np.pi),
                x2, y, p0=2., bounds=(0, np.inf))
            kappa = round(popt[0], 3)

            y_fit = self._uniform_vonmises_dist2(x, q, kappa)

            ax[j0, j1].plot(x2, y_fit, c=cols[str(ii)], label=r'q=%.3f, $\kappa$=%.3f' % (q, kappa))
            ax[j0, j1].set_ylim(0, 1)
            plotter.draw_legend(ax[j0, j1])
        plotter.save(fig, name=name + '_fit')
