from Events import Events


class EventsBuilder:

	@staticmethod
	def build(tab, name):
		return Events(tab, name)
