from DataObjects.Events1d import Events1dBuilder


class Filters:
	def __init__(self):
		pass

	@staticmethod
	def filter(obj, event, name, label=None, category=None, description=None):
		index = event.array.index
		array = obj.array.loc[index, :]
		return Events1dBuilder.build(
			array=array, name=name, category=category, label=label, description=description)
