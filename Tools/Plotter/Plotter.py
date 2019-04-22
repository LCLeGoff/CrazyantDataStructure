import os

import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
from matplotlib import colors

import Tools.MiscellaneousTools.ArrayManipulation as array_manip

from Tools.Plotter.BasePlotters import BasePlotters
from Tools.Plotter.FeatureArguments import ArgumentsTools, LineFeatureArguments, AxisFeatureArguments


class Plotter(BasePlotters):
    def __init__(self, root, obj, column_name=None, category=None, cmap='hot', **kwargs):

        if obj.category is None:
            if category is None:
                self.root = None
            else:
                self.root = root+category+'/Plots/'
        else:
            self.root = root+obj.category+'/Plots/'

        self.column_name = column_name

        BasePlotters.__init__(self, obj, cmap=cmap)

        self.arg_tools = ArgumentsTools(self)

        self.arg_tools.add_arguments('line', LineFeatureArguments())
        self.arg_tools.change_arg_value('line', kwargs)

        self.arg_tools.add_arguments('axis', AxisFeatureArguments())
        self.axis['xlabel'] = obj.label
        self.axis['ylabel'] = column_name
        self.arg_tools.change_arg_value('axis', kwargs)

    def plot(
            self, normed=False, title=None, title_prefix=None, label_suffix=None, label=None,
            preplot=None, figsize=None, **kwargs):

        fig, ax, label = self.__prepare_plot(preplot, figsize, label, label_suffix, title, title_prefix, kwargs)

        x = self.obj.get_index_array()

        y = self.__get_y(normed, x)
        self.__plot_xy(ax, x, y, label)

        return fig, ax

    def plot_smooth(self, window, normed=False, title=None, title_prefix=None, label_suffix=None, label=None,
                    preplot=None, figsize=None, **kwargs):

        fig, ax, label = self.__prepare_plot(preplot, figsize, label, label_suffix, title, title_prefix, kwargs)

        x = self.obj.get_index_array()
        y = self.__get_y(normed, x, window, smooth=True)
        self.__plot_xy(ax, x, y, label)

        return fig, ax

    def plot_heatmap(self, normed=False, title=None, title_prefix=None, preplot=None, figsize=None,
                     vmin=None, vmax=None, display_cbar=True, cmap_scale_log=False, colorbar_ticks=None,
                     cbar_ticks_size=15, cbar_label=None, cbar_label_size=15, nan_color='grey', **kwargs):

        fig, ax, label = self.__prepare_plot(preplot=preplot, figsize=figsize,
                                             title=title, title_prefix=title_prefix, kwargs=kwargs)

        tab_x = list(self.obj.get_index_array())
        tab_x.append(2*tab_x[-1]-tab_x[-2])
        tab_x = np.array(tab_x)

        tab_y = list(np.array(self.obj.get_column_names(), dtype=float))
        tab_y.append(2*tab_y[-1]-tab_y[-2])
        tab_y = np.array(tab_y)

        tab_h = self.obj.get_array()
        h = np.ma.array(tab_h, mask=np.isnan(tab_h))

        if h.size <= 1e5:

            y, x = np.meshgrid(tab_y + 1e-5 * (ax.get_yscale == 'log'), tab_x + 1e-5 * (ax.get_xscale == 'log'))
            if normed is True:
                s = np.sum(h)
                h = h/s
            elif normed == 'conditional_x':
                l1, l2 = h.shape
                for i in range(l2):
                    h[:, i] /= np.sum(h[:, i])
            elif normed == 'conditional_y':
                l1, l2 = h.shape
                for i in range(l1):
                    h[i, :] /= np.sum(h[i, :])

            if cmap_scale_log:
                img = ax.pcolor(x, y, h, vmin=vmin, vmax=vmax, norm=colors.LogNorm(), cmap=self.cmap)
            else:
                img = ax.pcolor(x, y, h, vmin=vmin, vmax=vmax, cmap=self.cmap)

            cmap = mlt.cm.get_cmap()
            cmap.set_bad(color=nan_color)

            if display_cbar:
                if colorbar_ticks is None:
                    cbar = fig.colorbar(img, ax=ax)
                else:
                    cbar = fig.colorbar(img, ticks=colorbar_ticks[0], ax=ax)
                    cbar.ax.set_yticklabels(colorbar_ticks[1], fontsize=cbar_ticks_size)

                cbar.set_label(cbar_label, fontsize=cbar_label_size, va='center', fontweight='bold')
            else:
                cbar = None

            return fig, ax, cbar

        else:
            print('Matrix too big: ' + str(h.size))

    def __prepare_plot(self, preplot=None, figsize=None, label=None, label_suffix=None,
                       title=None, title_prefix=None, kwargs=None):

        if label_suffix is None:
            label_suffix = ''
        else:
            label_suffix = ' ' + label_suffix

        if label is None:
            if self.obj.get_dimension() == 1 or self.column_name is not None:
                label = self.obj.label
            else:
                label = [str(column_name) + label_suffix for column_name in self.obj.get_column_names()]

        self.arg_tools.change_arg_value('line', kwargs)
        self.arg_tools.change_arg_value('axis', kwargs)
        fig, ax = self.create_plot(preplot, figsize=figsize)
        self.display_title(ax=ax, title_prefix=title_prefix, title=title)
        self.set_axis_scales_and_labels(ax, self.axis)
        return fig, ax, label

    def __plot_xy(self, ax, x, y, label):
        if self.obj.get_dimension() == 1:
            ax.plot(x, y, label=label, **self.line)
        else:
            if self.column_name is None:

                colors = self.color_object.create_cmap(self.cmap, self.obj.get_column_names())

                for i, column_name in enumerate(self.obj.get_column_names()):
                    self.line['c'] = colors[str(column_name)]
                    self.line['markeredgecolor'] = colors[str(column_name)]
                    ax.plot(x, y[i], label=label[i], **self.line)
                ax.legend(loc=0)
            else:
                if self.column_name not in self.obj.get_column_names():
                    self.column_name = str(self.column_name)
                ax.plot(x, y, label=label, **self.line)

    def __get_y(self, normed, x, window=None, smooth=False):

        if smooth is False:
            smooth = lambda w, q: w
        else:
            smooth = array_manip.smooth

        if self.obj.get_dimension() == 1:
            y = smooth(self.__norm_y(self.obj.get_array(), normed, x), window)
        else:
            if self.column_name is None:
                y = []
                for i, column_name in enumerate(self.obj.get_column_names()):
                    y.append(smooth(self.__norm_y(self.obj.df[column_name], normed, x), window))
            else:
                if self.column_name not in self.obj.get_column_names():
                    self.column_name = str(self.column_name)
                y = smooth(self.__norm_y(self.obj.df[self.column_name], normed, x), window)

        return y

    @staticmethod
    def __norm_y(y, normed, x):
        y0 = y.ravel()
        if normed is True:
            dx = float(np.mean(x[1:] - x[:-1]))
            s = float(sum(y0))
            y0 = y0.copy() / dx / s
        return y0

    def save(self, fig, name=None, suffix=None, sub_folder=None):
        if self.root is None:
            raise NameError(self.obj.name+' not properly defined')
        else:
            if name is None:
                name = self.obj.name
            if sub_folder is None:
                sub_folder = ''
            else:
                if sub_folder[-1] != '/':
                    sub_folder += '/'
                os.makedirs(self.root + sub_folder, exist_ok=True)

            if suffix is None:
                suffix = ''
            else:
                suffix = '_'+suffix

            address = self.root + sub_folder + str(name) + suffix + '.png'
            fig.savefig(address)
            fig.clf()
            plt.close()
