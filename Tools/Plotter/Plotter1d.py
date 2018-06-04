import numpy as np

from Tools.Plotter.BasePlotters import BasePlotters
from Tools.Plotter.ColorObject import ColorObject
from Tools.Plotter.FeatureArguments import ArgumentsTools, LineFeatureArguments
from Tools.Plotter.Plotter2d import Plotter2d


class Plotter1d(BasePlotters):
    def __init__(self, obj, **kwargs):
        BasePlotters.__init__(self, obj)
        self.arg_tools = ArgumentsTools(self)
        self.arg_tools.add_arguments('line', LineFeatureArguments())
        self.arg_tools.change_arg_value('line', kwargs)
        self.plotter2d = Plotter2d(self.obj)

    def hist1d(
            self, bins='fd', normed=False, label=None, xscale=None, yscale=None, multi_plot=None, title_prefix=None,
            preplot=None, **kwargs):
        if label is None:
            label = ''

        self.arg_tools.change_arg_value('line', kwargs)

        fig, ax = self.create_plot(preplot)
        self.axis_scale(ax, xscale, yscale)

        if multi_plot == 'exp':
            self._plot_hist_for_each_exp(ax, bins, normed)
        elif multi_plot == 'ant':
            self._plot_hist_for_each_ant(ax, bins, normed)
        else:
            self._plot_hist(ax, self.obj, bins, normed, label=label)
        self.display_title(ax, title_prefix)
        return fig, ax

    def _plot_hist_for_each_ant(self, ax, bins, normed):
        id_exp_ant_list = self.obj.get_index_array_of_id_exp_ant()
        col_list = ColorObject('cmap', self.cmap, id_exp_ant_list).colors
        for id_exp, id_ant in id_exp_ant_list:
            sub_obj = self.obj.get_row_of_id_exp_ant(id_exp, id_ant)
            self._plot_hist(ax, sub_obj, bins, normed, col_list[(id_exp, id_ant)])

    def _plot_hist_for_each_exp(self, ax, bins, normed):
        id_exp_list = self.obj.get_index_array_of_id_exp()
        col_list = ColorObject('cmap', self.cmap, id_exp_list).colors
        for id_exp in id_exp_list:
            sub_obj = self.obj.get_row_of_id_exp(id_exp)
            self._plot_hist(ax, sub_obj, bins, normed, col_list[id_exp])

    def _plot_hist(self, ax, obj, bins, normed, label, color=None):
        x, y = self._compute_histogram(obj, bins, normed)
        if color is not None:
            self.line['c'] = color
        ax.plot(x, y, '.-', **self.line, label=label)

    def _compute_histogram(self, sub_obj, bins, normed):
        y, x = np.histogram(sub_obj.df, bins, normed=normed)
        x = self.compute_x_inter(x)
        return x, y

    @staticmethod
    def compute_x_inter(x):
        x = (x[1:] + x[:-1]) / 2.
        return x

    def radial_direction_in_arena(self, preplot=None, **kwarg):
        self.plotter2d.radial_direction_in_arena(preplot=preplot, **kwarg)
