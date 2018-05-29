from matplotlib import pyplot as plt


class BasePlotters:
	def __init__(self, obj, cmap='jet', **kwargs):
		self.cmap = cmap
		self.obj = obj

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

	def create_plot(self, preplot=None, figsize=(5, 4)):
		if preplot is None:
			fig, ax = plt.subplots(figsize=figsize)
			self.black_background(fig, ax)
			return fig, ax
		else:
			return preplot

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

	def display_title(self, ax, title_prefix=None):
		if title_prefix is None:
			ax.set_title(self.obj.definition.label)
		else:
			print(title_prefix+': '+self.obj.definition.label)
			ax.set_title(title_prefix+': '+self.obj.definition.label)
