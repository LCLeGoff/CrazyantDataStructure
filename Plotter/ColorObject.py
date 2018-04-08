from matplotlib import colors
from matplotlib import pyplot as plt


class ColorObject:
	def __init__(self, kind, arg, idx_list):
		self.colors = dict()
		if kind == 'c' or kind == 'color':
			for idx in idx_list:
				self.colors[idx] = arg
		elif kind == 'cmap':
			norm = colors.Normalize(0, len(idx_list)-1)
			cmap = plt.get_cmap(arg)
			for i, idx in enumerate(idx_list):
				self.colors[idx] = cmap(norm(i))
