from JsonFiles import JsonFiles


class AnalyseStarter:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.group = group

	def start(self):
		ref_names = dict()

		for key in [
			'area', 'eccentricity',
			'major_axis_length', 'minor_axis_length', 'perimeter', 'markings'
		]:
			ref_names[key] = dict()
			ref_names[key]['label'] = key.capitalize()
			ref_names[key]['category'] = 'Raw'
			ref_names[key]['object_type'] = 'TimeSeries'

		for key in ['x', 'y']:
			key2 = key+'0'
			ref_names[key2] = dict()
			ref_names[key2]['label'] = key2
			ref_names[key2]['category'] = 'Raw'
			ref_names[key2]['object_type'] = 'TimeSeries'
			ref_names[key2]['description'] = key+' coordinate (in the cropped image system)'

		key = 'absoluteOrientation'
		ref_names[key] = dict()
		ref_names[key]['label'] = 'Absolute orientation'
		ref_names[key]['category'] = 'Raw'
		ref_names[key]['object_type'] = 'TimeSeries'
		ref_names[key]['description'] = 'absolute orientation'

		ref_names['area']['description'] = 'area'
		ref_names['eccentricity']['description'] = 'eccentricity'
		ref_names['major_axis_length']['description'] = 'major axis length'
		ref_names['minor_axis_length']['description'] = 'minor axis length'
		ref_names['major_axis_length']['label'] = 'major axis length'
		ref_names['minor_axis_length']['label'] = 'minor axis length'
		ref_names['perimeter']['description'] = 'perimeter'

		ref_names['markings']['description'] = 'Marking events'
		ref_names['markings']['object_type'] = 'Events'

		for key in [
			'session', 'trial', 'n_frames', 'fps', 'mm2px', 'food_radius',
			'food_center', 'traj_translation', 'crop_limit_x', 'crop_limit_y'
		]:
			ref_names[key] = dict()
			ref_names[key]['label'] = key.capitalize()
			ref_names[key]['category'] = 'Raw'
			ref_names[key]['object_type'] = 'Characteristics'

		ref_names['session']['description'] = 'trial number of the experiment'
		ref_names['trial']['description'] = 'session number of the experiment'
		ref_names['n_frames']['description'] = 'number of frames of the experiment'
		ref_names['n_frames']['label'] = 'Frame number'
		ref_names['fps']['description'] = 'frame per second of the movie'
		ref_names['mm2px']['description'] = 'ratio to convert millimeters to pixels'
		ref_names['mm2px']['label'] = 'mm to px'
		ref_names['food_radius']['description'] = 'radius of the food piece'
		ref_names['food_radius']['label'] = 'Food radius'
		ref_names['food_center']['description'] = 'coordinates of the center of the food piece'
		ref_names['food_center']['label'] = 'Food center'
		ref_names['traj_translation']['description'] = 'Translation vector between the picture and the cropped picture'
		ref_names['traj_translation']['label'] = 'Trajectory translation'
		ref_names['crop_limit_x']['description'] = 'limits of the crop on the x coordinates'
		ref_names['crop_limit_x']['label'] = 'Crop limit x'
		ref_names['crop_limit_y']['description'] = 'limits of the crop on the y coordinates'
		ref_names['crop_limit_y']['label'] = 'Crop limit y'
		for i in [1, 2]:
			key = 'entrance'+str(i)
			ref_names[key] = dict()
			ref_names[key]['label'] = 'Entrance point '+str(i)
			ref_names[key]['category'] = 'Raw'
			ref_names[key]['object_type'] = 'Characteristics'
			ref_names[key]['description'] = 'One of the two points delimiting the entrance'

		for i in [1, 2]:
			key = 'ref_pts'+str(i)
			ref_names[key] = dict()
			ref_names[key]['label'] = key.capitalize()
			ref_names[key]['category'] = 'Raw'
			ref_names[key]['object_type'] = 'Characteristics'
			ref_names[key]['description'] = 'One of the two reference points'
			ref_names[key]['label'] = 'Reference points '+str(i)

		JsonFiles.write_obj(self.root+'ref_names.json', ref_names)
