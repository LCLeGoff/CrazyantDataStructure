from JsonFiles import JsonFiles


class AnalyseStarter:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.group = group

	def start(self):
		ref_names = dict()

		for key in [
			'x', 'y', 'orientation', 'area', 'eccentricity', 'major_axis_length', 'minor_axis_length', 'perimeter'
		]:
			ref_names[key] = dict()
			ref_names[key]['label'] = key
			ref_names[key]['category'] = 'raw'
			ref_names[key]['object_type'] = 'TimeSeries'

		ref_names['major_axis_length']['label'] = 'major axis length'
		ref_names['minor_axis_length']['label'] = 'minor axis length'

		end_txt = ' of the blob number "id_ant" at frame number "frame"'
		ref_names['x']['description'] = 'x coordinate (in the cropped image)'+end_txt
		ref_names['y']['description'] = 'y coordinate (in the cropped image)'+end_txt
		ref_names['orientation']['description'] = 'absolute orientation'+end_txt
		ref_names['area']['description'] = 'area'+end_txt
		ref_names['eccentricity']['description'] = 'eccentricity'+end_txt
		ref_names['major_axis_length']['description'] = 'major axis length'+end_txt
		ref_names['minor_axis_length']['description'] = 'minor axis length'+end_txt
		ref_names['perimeter']['description'] = 'perimeter'+end_txt

		JsonFiles.write_obj(self.root+'ref_names.json', ref_names)
