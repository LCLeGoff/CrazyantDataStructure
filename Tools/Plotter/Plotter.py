import numpy as np

from Tools.Plotter.BasePlotters import BasePlotters
from Tools.Plotter.FeatureArguments import ArgumentsTools, LineFeatureArguments, AxisFeatureArguments
from Tools.Plotter.ColorObject import ColorObject


class Plotter(BasePlotters):
    def __init__(self, root, obj, column_name=None, **kwargs):

        if obj.category is not None:
            self.root = root+obj.category+'/Plots/'+obj.name
        else:
            self.root = None

        if obj.get_nbr_index() == 1:
            if column_name is None:
                column_name = obj.get_column_names()[0]

            if column_name in obj.get_column_names():

                BasePlotters.__init__(self, obj)

                self.arg_tools = ArgumentsTools(self)

                self.arg_tools.add_arguments('line', LineFeatureArguments())
                self.arg_tools.change_arg_value('line', kwargs)

                self.arg_tools.add_arguments('axis', AxisFeatureArguments())
                self.axis['xlabel'] = obj.label
                self.axis['ylabel'] = column_name
                self.arg_tools.change_arg_value('axis', kwargs)
            else:
                raise NameError(column_name+' is not a column name')

        else:
            raise IndexError('There are more than one index columns')

    def plot(
            self, normed=False, title=None, title_prefix=None, label_suffix=None,
            preplot=None, **kwargs):
        if label_suffix is None:
            label_suffix = ''
        else:
            label_suffix = ' '+label_suffix

        self.arg_tools.change_arg_value('line', kwargs)
        self.arg_tools.change_arg_value('axis', kwargs)

        fig, ax = self.create_plot(preplot)
        self.display_title(ax=ax, title_prefix=title_prefix, title=title)
        self.set_axis_scales_and_labels(ax, self.axis)

        x = self.obj.get_index_array()

        if self.obj.get_dimension() == 1:
            y = self.__get_y(self.obj.get_array(), normed, x)
            ax.plot(x, y, **self.line)

        else:
            colors = ColorObject.create_cmap(self.cmap, self.obj.get_column_names())
            for column_name in self.obj.get_column_names():
                y = self.__get_y(self.obj.df[column_name], normed, x)
                self.line['c'] = colors[str(column_name)]
                ax.plot(x, y, label=str(column_name)+label_suffix, **self.line)
            ax.legend(loc=0)

        return fig, ax

    @staticmethod
    def __get_y(y, normed, x):
        y0 = y.ravel()
        if normed is True:
            dx = float(np.mean(x[1:] - x[:-1]))
            s = float(sum(y0))
            y0 = y0.copy() / dx / s
        return y0

    def save(self, fig, suffix=None):
        if self.root is None:
            raise NameError(self.obj.name+'not properly defined')
        else:
            if suffix is None:
                fig.savefig(self.root+'.png')
            else:
                fig.savefig(self.root+suffix+'.png')
            fig.clf()
