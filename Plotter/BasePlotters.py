from matplotlib import pyplot as plt


class BasePlotters:
	def __init__(self, **kwargs):
		default_args = {
			"ls": '-',
			"lw": 3,
			"c": ('c', 'w'),
			"alpha": 1,
			"ms": 5
		}
		self.__dict__.update(default_args)
		for (arg_name, default_value) in default_args.items():
			if arg_name in default_args:
				setattr(self, arg_name, kwargs.get(arg_name, default_value))
			else:
				raise TypeError('Plotters does not have '+arg_name+' as argument')

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

	def create_plot(self, figsize):
		fig, ax = plt.subplots(figsize=figsize)
		self.black_background(fig, ax)
		return fig, ax

	@staticmethod
	def remove_axis(fig, ax):
		ax.axis('off')
		fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)
