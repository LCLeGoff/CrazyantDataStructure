from matplotlib import pyplot as plt

from Tools.Plotter.ColorObject import ColorObject


class BasePlotters:
    def __init__(self, obj=None, cmap='jet_r', **kwargs):
        self.cmap = cmap
        self.obj = obj
        self.color_object = ColorObject

    @staticmethod
    def black_background(fig, ax):
        bg_color = 'black'
        fg_color = 'white'
        fig.set_facecolor(bg_color)
        ax.patch.set_facecolor(bg_color)
        ax.spines['right'].set_color(fg_color)
        ax.spines['left'].set_color(fg_color)
        ax.spines['bottom'].set_color(fg_color)
        ax.spines['top'].set_color(fg_color)
        ax.xaxis.set_tick_params(color=fg_color, labelcolor=fg_color, which='both')
        ax.yaxis.set_tick_params(color=fg_color, labelcolor=fg_color, which='both')
        ax.xaxis.label.set_color(fg_color)
        ax.yaxis.label.set_color(fg_color)
        ax.title.set_color(fg_color)

    @staticmethod
    def grey_background(fig, ax):
        bg_color = '0.8'
        fig.set_facecolor(bg_color)
        ax.patch.set_facecolor(bg_color)

    def create_plot(self, preplot=None, figsize=(8, 8),
                    left=0.13, right=0.98, bottom=0.1, top=0.95, wspace=0.2, hspace=0.2,
                    nrows=1, ncols=1):
        if preplot is None:
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
            # self.black_background(fig, ax)
            if nrows == 1:
                if ncols == 1:
                    self.grey_background(fig, ax)
                else:
                    for ax0 in ax:
                        self.grey_background(fig, ax0)
            else:
                if ncols == 1:
                    for ax0 in ax:
                        self.grey_background(fig, ax0)
                else:
                    for i in range(ax.shape[0]):
                        for j in range(ax.shape[1]):
                            self.grey_background(fig, ax[i, j])

            return fig, ax
        else:
            return preplot

    @staticmethod
    def set_axis_scales_and_labels(ax, axis_dict):
        if axis_dict['xscale'] is None:
            ax.set_xscale('linear')
        else:
            ax.set_xscale(axis_dict['xscale'])
        if axis_dict['yscale'] is None:
            ax.set_yscale('linear')
        else:
            ax.set_yscale(axis_dict['yscale'])

        if axis_dict['xlabel'] is not None:
            ax.set_xlabel(axis_dict['xlabel'])
        if axis_dict['ylabel'] is not None:
            ax.set_ylabel(axis_dict['ylabel'])

    @staticmethod
    def remove_axis(fig, ax):
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)

    def display_title(self, ax, title_prefix=None, title=None):
        if title is None:
            title = self.obj.definition.label
        if title_prefix is None:
            ax.set_title(title)
        else:
            if title is None:
                ax.set_title(title_prefix)
            else:
                ax.set_title(title_prefix + ' ' + title)

    @staticmethod
    def draw_horizontal_line(ax, val=0, c='k', ls='--'):
        ax.axhline(val, ls=ls, c=c)

    @staticmethod
    def draw_vertical_line(ax, val=0, c='k', ls='--'):
        ax.axvline(val, ls=ls, c=c)

