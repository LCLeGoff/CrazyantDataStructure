import numpy as np

from Tools.Plotter.BasePlotters import BasePlotters
from Tools.Plotter.FeatureArguments import ArgumentsTools, LineFeatureArguments, AxisFeatureArguments


class Plotter(BasePlotters):
    def __init__(self, root, obj, column_name=None, **kwargs):

        if obj.category is not None:
            self.root = root+'/'+obj.category+'/Plots/'+obj.name
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
            self, normed=False, title=None, title_prefix=None,
            preplot=None, **kwargs):

        self.arg_tools.change_arg_value('line', kwargs)
        self.arg_tools.change_arg_value('axis', kwargs)

        fig, ax = self.create_plot(preplot)
        self.display_title(ax=ax, title_prefix=title_prefix, title=title)
        self.set_axis_scales_and_labels(ax, self.axis)

        x = self.obj.get_index_array()
        y = self.obj.get_array().ravel()

        if normed is True:
            dx = float(np.mean(x[1:] - x[:-1]))
            s = float(sum(y))
            y = y.copy() / dx / s

        ax.plot(x, y, **self.line)
        return fig, ax

    def save(self, fig, suffix=None):
        if self.root is None:
            raise NameError(self.obj.name+'not properly defined')
        else:
            if suffix is None:
                fig.savefig(self.root+'.png')
            else:
                fig.savefig(self.root+suffix+'.png')
            fig.clf()
