import numpy as np
from Plotter.ColorObject import ColorObject
from Plotter.BasePlotters import BasePlotters
import matplotlib.pyplot as plt


class Plotter1d(BasePlotters):
	def __init__(self, **kwargs):
		BasePlotters.__init__(self, **kwargs)

	@staticmethod
	def check_type(obj):
		if obj.object_type[-2:] == '2d':
			raise TypeError(obj.name+' is not a 1d data')

	def hist1d(self, obj, bins, xscale=None, yscale=None, group=0):
		self.check_type(obj)

		if group == 0:
			fig, ax = self.create_plot()
			self.axis_scale(ax, xscale, yscale)
			col = ColorObject(self.c[0], self.c[1]).colors
			y, x = np.histogram(obj.df, bins)
			x = (x[1:]+x[:-1])/2.
			ax.plot(x, y, '.-', ms=self.ms, alpha=self.alpha, color=col)
		elif group == 1:
			fig, ax = self.create_plot()
			self.axis_scale(ax, xscale, yscale)
			list_exp = obj.get_index_array_of_id_exp()
			cols = ColorObject('cmap', 'jet', list_exp).colors
			for id_exp in list_exp:
				tab = obj.get_row_of_id_exp(id_exp)
				y, x = np.histogram(tab, bins)
				x = (x[1:]+x[:-1])/2.
				ax.plot(x, y, '.-', ms=self.ms, alpha=self.alpha, color=cols[id_exp])
		elif group == 2:
			list_exp = obj.get_index_array_of_id_exp()
			cols = ColorObject('cmap', 'jet', list_exp).colors
			for id_exp in list_exp:
				fig, ax = self.create_plot()
				self.axis_scale(ax, xscale, yscale)
				tab = obj.get_row_of_id_exp(id_exp)
				y, x = np.histogram(tab, bins)
				x = (x[1:]+x[:-1])/2.
				ax.plot(x, y, '.-', ms=self.ms, alpha=self.alpha, color=cols[id_exp])
		elif group == 3:
			list_exp_ant_dict = obj.get_index_dict_of_id_exp_ant()
			for id_exp in list_exp_ant_dict:
				fig, ax = self.create_plot()
				self.axis_scale(ax, xscale, yscale)
				cols = ColorObject('cmap', 'jet', list_exp_ant_dict[id_exp]).colors
				for id_ant in list_exp_ant_dict[id_exp]:
					tab = obj.get_row_of_id_exp_ant(id_exp, id_ant)
					y, x = np.histogram(tab, bins)
					x = (x[1:]+x[:-1])/2.
					ax.plot(x, y, '.-', ms=self.ms, alpha=self.alpha, color=cols[id_ant])
				plt.show()

