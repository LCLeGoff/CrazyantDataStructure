from JsonFiles import JsonFiles


class AnalyseStarter:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.group = group

	def start(self):
		ref_names = dict()

		for key in [
			'x', 'y', 'orientation', 'area', 'eccentricity',
			'major_axis_length', 'minor_axis_length', 'perimeter', 'markings'
		]:
			ref_names[key] = dict()
			ref_names[key]['label'] = key.capitalize()
			ref_names[key]['category'] = 'Raw'
			ref_names[key]['object_type'] = 'TimeSeries'

		ref_names['x']['description'] = 'x coordinate (in the cropped image)'
		ref_names['x']['label'] = 'x'
		ref_names['y']['description'] = 'y coordinate (in the cropped image)'
		ref_names['y']['label'] = 'y'
		ref_names['orientation']['description'] = 'absolute orientation'
		ref_names['area']['description'] = 'area'
		ref_names['eccentricity']['description'] = 'eccentricity'
		ref_names['major_axis_length']['description'] = 'major axis length'
		ref_names['minor_axis_length']['description'] = 'minor axis length'
		ref_names['major_axis_length']['label'] = 'major axis length'
		ref_names['minor_axis_length']['label'] = 'minor axis length'
		ref_names['perimeter']['description'] = 'perimeter'

		ref_names['markings']['description'] = 'Marking events'
		ref_names['markings']['object_type'] = 'Events'

		JsonFiles.write_obj(self.root+'ref_names.json', ref_names)
