from matplotlib import pyplot as plt
import numpy as np
from Plotter.ColorObject import ColorObject
from Plotter.BasePlotters import BasePlotters


class Plotter2d(BasePlotters):
	def __init__(self, **kwargs):
		BasePlotters.__init__(self, ls='o', c=('cmap', 'hot'), alpha=0.6, **kwargs)
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

	def repartition(self, obj, title_prefix=''):
		self.check_type(obj)

		mark_indexes = obj.get_array_id_exp()
		cols = ColorObject(self.c[0], self.c[1], mark_indexes).colors
		fig, ax = self.create_plot((6.5, 5))
		self.draw_setup(fig, ax)
		self.display_title(ax, obj, title_prefix)

		for id_exp in mark_indexes:
			ax.plot(
				obj.get_x_values().loc[id_exp, :, :], obj.get_y_values().loc[id_exp, :, :],
				self.ls, ms=self.ms, alpha=self.alpha, color=cols[id_exp])

		return fig, ax

	def hist2d(self, obj, bins=100, normed=False, title_prefix=''):

		fig, ax = self.create_plot((9, 5))
		self.display_labels(ax, obj)
		self.display_title(ax, obj, title_prefix)
		self.draw_setup(fig, ax)
		plt.hist2d(obj.get_x_values(), obj.get_y_values(), bins=bins, cmap=self.c[1], normed=normed)
		plt.colorbar()
		plt.axis('equal')

		return fig, ax
