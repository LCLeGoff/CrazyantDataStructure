from DataObjects.Events import EventsBuilder


class Filters:
	def __init__(self):
		pass

	@staticmethod
	def filter(obj, event, name, label, category, description):
		index = event.array.index
		array = obj.array.loc[index, :]
		return EventsBuilder.build(array, name, category, label, description)
