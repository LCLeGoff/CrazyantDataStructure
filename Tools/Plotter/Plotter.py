import os

import numpy as np
import matplotlib.pyplot as plt

from Tools.Plotter.BasePlotters import BasePlotters
from Tools.Plotter.FeatureArguments import ArgumentsTools, LineFeatureArguments, AxisFeatureArguments


class Plotter(BasePlotters):
    def __init__(self, root, obj, column_name=None, category=None, **kwargs):

        if obj.category is None:
            if category is None:
                self.root = None
            else:
                self.root = root+category+'/Plots/'
        else:
            self.root = root+obj.category+'/Plots/'

        self.column_name = column_name

        BasePlotters.__init__(self, obj)

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
        if label_suffix is None:
            label_suffix = ''
        else:
            label_suffix = ' '+label_suffix

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

        x = self.obj.get_index_array()

        if self.obj.get_dimension() == 1:
            y = self.__get_y(self.obj.get_array(), normed, x)
            ax.plot(x, y, label=label, **self.line)
        else:
            if self.column_name is None:

                colors = self.color_object.create_cmap(self.cmap, self.obj.get_column_names())

                for i, column_name in enumerate(self.obj.get_column_names()):
                    y = self.__get_y(self.obj.df[column_name], normed, x)
                    self.line['c'] = colors[str(column_name)]
                    self.line['markeredgecolor'] = colors[str(column_name)]
                    ax.plot(x, y, label=label[i], **self.line)
                ax.legend(loc=0)
            else:
                if self.column_name not in self.obj.get_column_names():
                    self.column_name = str(self.column_name)
                y = self.__get_y(self.obj.df[self.column_name], normed, x)
                ax.plot(x, y, label=label, **self.line)

        return fig, ax

    @staticmethod
    def __get_y(y, normed, x):
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
