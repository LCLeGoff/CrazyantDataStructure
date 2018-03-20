import matplotlib.pyplot as plt


class Data2d:

	def __init__(self, xy):
		self.xy = xy

	def plot(self):
		ax = self.xy.plot('x', 'y', style='o')
		ax.axis('equal')
		plt.show()
