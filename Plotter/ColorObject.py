from matplotlib import colors
from matplotlib import pyplot as plt


class ColorObject:
	def __init__(self, kind, arg, idx_list=None):
		self.colors = dict()
		if kind == 'c' or kind == 'color':
			if idx_list is None:
				self.colors = arg
			else:
				for idx in idx_list:
					self.colors[idx] = arg

		elif kind == 'cmap':
			self.colors = self.create_cmap(arg, idx_list)

	@staticmethod
	def create_cmap(cmap, idx_list):
		cols = dict()
		if isinstance(idx_list, int) or isinstance(idx_list, float):
			norm = colors.Normalize(0, int(idx_list)-1)
			cmap = plt.get_cmap(cmap)
			for i in range(int(idx_list)):
				cols[i] = cmap(norm(i))
		else:
			norm = colors.Normalize(0, len(idx_list)-1)
			cmap = plt.get_cmap(cmap)
			for i, idx in enumerate(idx_list):
				cols[idx] = cmap(norm(i))
		return cols
