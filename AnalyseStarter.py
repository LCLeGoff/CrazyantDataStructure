from Tools.JsonFiles import write_obj


class AnalyseStarter:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.group = group

	def start(self):
		def_dict = dict()

		for key in [
			'area', 'eccentricity',
			'major_axis_length', 'minor_axis_length', 'perimeter', 'markings'
		]:
			def_dict[key] = dict()
			def_dict[key]['label'] = key.capitalize()
			def_dict[key]['category'] = 'Raw'
			def_dict[key]['object_type'] = 'TimeSeries'

		for key in ['x', 'y']:
			key2 = key+'0'
			def_dict[key2] = dict()
			def_dict[key2]['label'] = key2
			def_dict[key2]['category'] = 'Raw'
			def_dict[key2]['object_type'] = 'TimeSeries'
			def_dict[key2]['description'] = key+' coordinate (in the cropped image system)'

		key = 'absoluteOrientation'
		def_dict[key] = dict()
		def_dict[key]['label'] = 'Absolute orientation'
		def_dict[key]['category'] = 'Raw'
		def_dict[key]['object_type'] = 'TimeSeries'
		def_dict[key]['description'] = 'absolute orientation'

		def_dict['area']['description'] = 'area'
		def_dict['eccentricity']['description'] = 'eccentricity'
		def_dict['major_axis_length']['description'] = 'major axis length'
		def_dict['minor_axis_length']['description'] = 'minor axis length'
		def_dict['major_axis_length']['label'] = 'major axis length'
		def_dict['minor_axis_length']['label'] = 'minor axis length'
		def_dict['perimeter']['description'] = 'perimeter'

		def_dict['markings']['description'] = 'Marking events'
		def_dict['markings']['object_type'] = 'Events'

		for key in [
			'session', 'trial', 'n_frames', 'fps', 'mm2px', 'food_radius',
			'food_center', 'traj_translation', 'crop_limit_x', 'crop_limit_y'
		]:
			def_dict[key] = dict()
			def_dict[key]['label'] = key.capitalize()
			def_dict[key]['category'] = 'Raw'
			def_dict[key]['object_type'] = 'Characteristics1d'

		def_dict['session']['description'] = 'trial number of the experiment'
		def_dict['trial']['description'] = 'session number of the experiment'
		def_dict['n_frames']['description'] = 'number of frames of the experiment'
		def_dict['n_frames']['label'] = 'Frame number'
		def_dict['fps']['description'] = 'frame per second of the movie'
		def_dict['mm2px']['description'] = 'ratio to convert millimeters to pixels'
		def_dict['mm2px']['label'] = 'mm to px'
		def_dict['food_radius']['description'] = 'radius of the food piece'
		def_dict['food_radius']['label'] = 'Food radius'
		def_dict['food_center']['description'] = 'coordinates of the center of the food piece'
		def_dict['food_center']['label'] = 'Food center'
		def_dict['food_center']['object_type'] = 'Characteristics2d'
		def_dict['traj_translation']['description'] = 'Translation vector between the picture and the cropped picture'
		def_dict['traj_translation']['label'] = 'Trajectory translation'
		def_dict['traj_translation']['object_type'] = 'Characteristics2d'
		def_dict['crop_limit_x']['description'] = 'limits of the crop on the x coordinates'
		def_dict['crop_limit_x']['label'] = 'Crop limit x'
		def_dict['crop_limit_x']['object_type'] = 'Characteristics2d'
		def_dict['crop_limit_y']['description'] = 'limits of the crop on the y coordinates'
		def_dict['crop_limit_y']['label'] = 'Crop limit y'
		def_dict['crop_limit_y']['object_type'] = 'Characteristics2d'
		for i in [1, 2]:
			key = 'entrance'+str(i)
			def_dict[key] = dict()
			def_dict[key]['label'] = 'Entrance point '+str(i)
			def_dict[key]['category'] = 'Raw'
			def_dict[key]['object_type'] = 'Characteristics2d'
			def_dict[key]['description'] = 'One of the two points delimiting the entrance'

		for i in [1, 2]:
			key = 'ref_pts'+str(i)
			def_dict[key] = dict()
			def_dict[key]['label'] = key.capitalize()
			def_dict[key]['category'] = 'Raw'
			def_dict[key]['object_type'] = 'Characteristics2d'
			def_dict[key]['description'] = 'One of the two reference points'
			def_dict[key]['label'] = 'Reference points '+str(i)

		write_obj(self.root+'definition_dict.json', def_dict)
