from matplotlib import pyplot as plt

from Plotter.LineFeatureArguments import LineFeatureArguments


class BasePlotters:
	def __init__(self, **kwargs):
		self.line = LineFeatureArguments()
		for arg_name in self.line.__dict__:
			if arg_name in kwargs:
				self.line.__dict__[arg_name] = kwargs[arg_name]

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
		ax.title.set_color('w')

	def create_plot(self, figsize=(5, 4)):
		fig, ax = plt.subplots(figsize=figsize)
		self.black_background(fig, ax)
		return fig, ax

	@staticmethod
	def axis_scale(ax, xscale=None, yscale=None):
		if xscale is None:
			ax.set_xscale('linear')
		else:
			ax.set_xscale(xscale)
		if yscale is None:
			ax.set_yscale('linear')
		else:
			ax.set_yscale(yscale)

	@staticmethod
	def remove_axis(fig, ax):
		ax.axis('off')
		fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)

	@staticmethod
	def display_title(ax, obj, title_prefix):
		ax.set_title(title_prefix+': '+obj.definition.label)
