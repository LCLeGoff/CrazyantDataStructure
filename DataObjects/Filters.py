from DataObjects.Events import EventsBuilder


class Filters:
	def __init__(self):
		pass

	@staticmethod
	def filter(obj, event, name, label=None, category=None, description=None):
		index = event.array.index
		array = obj.array.loc[index, :]
		return EventsBuilder.build(
			array=array, name=name, category=category, label=label, description=description)
