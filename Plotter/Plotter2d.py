from matplotlib import pyplot as plt
import numpy as np
from Plotter.ColorObject import ColorObject


class Plotter2d:
	def __init__(self):
		self.food_color = 'paleturquoise'
		self.arena_color = 'paleturquoise'
		self.arena_length = 420.
		self.arena_width = 297.
		self.gate_length = 90.
		self.food_radius = 5.
		self.food_location = [0, 0]

	@staticmethod
	def check_type(obj):
		if obj.object_type[-2:] != '2d':
			raise TypeError(obj.name+' is not a 2d data')

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

	@staticmethod
	def init_plot_arg(c=None, ls=None, title_prefix=None):
		if ls is None:
			ls = 'o'
		if title_prefix is None:
			title_prefix = ''
		if c is None:
			c = ('cmap', 'hot')
		return c, ls, title_prefix

	@staticmethod
	def make_circle(radius=5., center=None):
		if center is None:
			center = [0, 0]
		theta = np.arange(-np.pi, np.pi+0.1, 0.1)
		return radius*np.cos(theta)+center[0], radius*np.sin(theta)+center[1]

	def draw_food(self, ax, lw=5):
		x, y = self.make_circle(self.food_radius, self.food_location)
		ax.plot(x, y, c=self.food_color, lw=lw)

	def draw_arena(self, ax, lw=5):
		ax.plot(
			[self.food_location[0]-self.arena_length/2., self.food_location[0]-self.arena_length/2.],
			[self.food_location[1]-self.arena_width/2., self.food_location[1]+self.arena_width/2.],
			color=self.arena_color, lw=lw
		)
		ax.plot(
			[self.food_location[0]+self.arena_length/2., self.food_location[0]+self.arena_length/2.],
			[self.food_location[1]-self.arena_width/2., self.food_location[1]+self.arena_width/2.],
			color=self.arena_color, lw=lw
		)
		ax.plot(
			[self.food_location[0]-self.arena_length/2., self.food_location[0]+self.arena_length/2.],
			[self.food_location[1]-self.arena_width/2., self.food_location[1]-self.arena_width/2.],
			color=self.arena_color, lw=lw
		)
		ax.plot(
			[self.food_location[0]-self.arena_length/2., self.food_location[0]+self.arena_length/2.],
			[self.food_location[1]+self.arena_width/2., self.food_location[1]+self.arena_width/2.],
			color=self.arena_color, lw=lw
		)

	def draw_gate(self, ax, lw=10):
		ax.plot(
			[self.food_location[0]+self.arena_length/2., self.food_location[0]+self.arena_length/2.],
			[self.food_location[1]-self.gate_length/2., self.food_location[1]+self.gate_length/2.],
			color='k', lw=lw
		)

	def draw_setup(self, fig, ax):
		ax.axis('equal')
		self.remove_axis(fig, ax)
		self.draw_food(ax)
		self.draw_arena(ax)
		self.draw_gate(ax)

	@staticmethod
	def display_labels(ax, obj):
		ax.set_xlabel(obj.definition.xlabel)
		ax.set_xlabel(obj.definition.xlabel)

	@staticmethod
	def display_title(ax, obj, title_prefix):
		ax.set_title(title_prefix+': '+obj.definition.label)

	def repartition(self, obj, c=None, ls=None, title_prefix=None, alpha=0.6):
		self.check_type(obj)
		c, ls, title_prefix = self.init_plot_arg(c, ls, title_prefix)

		mark_indexes = obj.get_id_exp_index()
		cols = ColorObject(c[0], c[1], mark_indexes).colors
		fig, ax = self.create_plot((6.5, 5))
		self.draw_setup(fig, ax)
		self.display_title(ax, obj, title_prefix)

		for id_exp in mark_indexes:
			ax.plot(
				obj.get_x_values().loc[id_exp, :, :], obj.get_y_values().loc[id_exp, :, :],
				ls, ms=3, alpha=alpha, color=cols[id_exp])

		return ax

	def hist2d(self, obj, bins=100, cmap='jet', normed=False, title_prefix=''):

		fig, ax = self.create_plot((9, 5))
		self.display_labels(ax, obj)
		self.display_title(ax, obj, title_prefix)
		plt.hist2d(obj.get_x_values(), obj.get_y_values(), bins=bins, cmap=cmap, normed=normed)
		plt.colorbar()
		plt.axis('equal')
