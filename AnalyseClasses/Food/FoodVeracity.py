import numpy as np
import pandas as pd
import Tools.MiscellaneousTools.Geometry as Geo
import scipy.stats as scs

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_frame_name
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

            self.exp.operation(name, lambda a: np.abs(a))
            self.exp.hist1d_evolution(name_to_hist=name, start_index_intervals=start_frame_intervals,
                                      end_index_intervals=end_frame_intervals, bins=bins,
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
        self.exp.operation(name, lambda a: np.abs(a))
        if redo:

            self.exp.hist1d_evolution(name_to_hist=name, start_index_intervals=start_frame_intervals,
                                      end_index_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category,
                                      label=result_label, description=result_description)

            self.exp.write(result_name)
            self.exp.remove_object(name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta$ (rad)', ylabel='PDF', normed=True, label_suffix=' s', title='')
        plotter.draw_legend(ax=ax, ncol=2)
        ax.set_ylim((0, 1))
        plotter.save(fig)
        temp_name = 'temp'

        self.exp.hist1d_evolution(name_to_hist=name, start_index_intervals=[45*100],
                                  end_index_intervals=[150*100], bins=bins,
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

        result_name = self.exp.hist1d_evolution(name_to_hist=name, start_index_intervals=start_frame_intervals,
                                                end_index_intervals=end_frame_intervals, bins=bins,
                                                category=self.category, normed=True)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.create_plot()
        columns = self.exp.get_columns(result_name)
        cols = plotter.color_object.create_cmap('hot', columns)
        for column in columns:
            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name=column)
            plotter.plot(
                preplot=(fig, ax), xlabel=r'$\theta$ (rad)', ylabel='PDF',
                c=cols[column], marker='.', ls='', label=r'$t\in$'+column+' s', display_legend=False)
            plotter.plot_fit(preplot=(fig, ax), typ='cst center gauss', c=cols[column], display_legend=False)
        plotter.save(fig, name=result_name+'_fit')

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.create_plot(figsize=(8, 8), nrows=3, ncols=3, bottom=0.06, top=0.96, hspace=0.4)
        columns = self.exp.get_columns(result_name)
        cols = plotter.color_object.create_cmap('hot', columns)
        n = int(5e5)
        for k, column in enumerate(columns):
            i = int(k/3)
            j = k-i*3

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name=column)
            plotter.plot(
                preplot=(fig, ax[i, j]), xlabel=r'$\theta$ (rad)', ylabel='PDF',
                c=cols[column], marker='.', ls='', title=r'$t\in$'+column+' s', display_legend=False)
            b, a, c, _, _ = plotter.plot_fit(
                preplot=(fig, ax[i, j]), typ='cst center gauss', c=cols[column], display_legend=False)

            # print(column, c, 1-np.pi*c)
            # if c < 1/np.pi:
            #     q = int(n*(1-np.pi*c))
            # else:
            #     q = 0
            # u = list(np.angle(np.exp(-1j*np.random.normal(scale=np.sqrt(0.8), size=q))))
            # v = list(np.pi-2*np.pi*np.random.uniform(size=n-q))
            # y, x = np.histogram(np.abs(u+v), np.arange(0, np.pi, 0.1), density=True)
            # ax[i, j].plot(x[:-1], y)

            ax[i, j].set_ylim(0, .8)

        plotter.save(fig, name=result_name+'_fit2')

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.create_plot(figsize=(8, 8), nrows=3, ncols=3, bottom=0.06, top=0.96, hspace=0.4)
        columns = self.exp.get_columns(result_name)
        cols = plotter.color_object.create_cmap('hot', columns)
        for k, column in enumerate(columns):
            i = int(k/3)
            j = k-i*3
            print(column)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name=column)
            plotter.plot(
                preplot=(fig, ax[i, j]), xlabel=r'$\theta$ (rad)', ylabel='PDF',
                c=cols[column], marker='.', ls='', title=r'$t\in$'+column+' s', display_legend=False)
            plotter.plot_fit(
                preplot=(fig, ax[i, j]), typ='cst vonmises', c=cols[column], display_legend=False)

            ax[i, j].set_ylim(0, .8)

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
        fig, ax = plotter.create_plot(figsize=(8, 8), nrows=3, ncols=3, bottom=0.06, top=0.96, hspace=0.4)
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
                ax[j0, j1].plot(x, y, 'o', c=cols[str(frame0)], mec='k')

                kappa, _, _ = scs.vonmises.fit(vals)
                ax[j0, j1].plot(x, scs.vonmises.pdf(x, kappa=kappa), c=cols[str(frame0)])
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
        a = Fits.exp_fct(x, a_s, b_s, c_s)
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

    def compute_fisher_info_evol_around_first_attachment(self, redo=False):
        name = 'food_direction_error_var_evol_around_first_attachment'
        result_name = 'fisher_info_evol_around_first_attachment'

        label = 'Fisher information over time'
        description = 'Fisher information over time (inverse of the variance of the food direction error)'

        if redo:
            self.exp.load(name)
            self.exp.add_copy(old_name=name, new_name=result_name, category=self.category,
                              label=label, description=description)
            self.exp.operation(result_name, lambda a: 1/a)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel='Time (s)', ylabel=r'Fisher information (rad$^{-2}$)',
                               label_suffix='s', label=r'Fisher information', marker='', title='')
        plotter.plot_fit(typ='exp', preplot=(fig, ax), window=[90, 400], cst=(-1, 1, 1))
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
