import pandas as pd

from Characteristics import Characteristics
from EventsBuilder import EventsBuilder
from JsonFiles import JsonFiles
from NameBuilder import NameBuilder
from TimeSeriesBuilder import TimeSeriesBuilder


class DataLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.name_builder = NameBuilder(root, group)
		self.time_series_loader = TimeSeriesLoader(root, group)
		self.events_loader = EventsLoader(root, group)
		self.characteristics_builder = CharacteristicsLoader(root, group)

	def load(self, name):
		name_class = self.name_builder.build(name)
		if name_class.object_type == 'TimeSeries':
			res = self.time_series_loader.load(name_class)
		elif name_class.object_type == 'Events':
			res = self.events_loader.load(name_class)
		elif name_class.object_type == 'Characteristics':
			res = self.characteristics_builder.build(name_class)
		else:
			print(name+' has no defined object type')
			res = None
		return res


class TimeSeriesLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'
		self.categories = dict()

	def load(self, name):
		add = self.root+name.category+'/TimeSeries.csv'
		if not(name.category in self.categories.keys()):
			self.categories[name.category] = pd.read_csv(add, index_col=['id_exp', 'id_ant', 'frame'])
		return TimeSeriesBuilder.build(self.categories[name.category][name.name], name)


class EventsLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'

	def load(self, name):
		add = self.root+name.category+'/'+name.name+'.csv'
		return EventsBuilder.build(pd.read_csv(add, index_col=['id_exp', 'id_ant', 'frame']), name)


class CharacteristicsLoader:
	def __init__(self, root, group):
		self.root = root+group+'/'

	def build(self, name):
		chara_dict = JsonFiles.import_obj(self.root+name.category+'/Characteristics.json')
		if name.name in ['session', 'trial', 'n_frames', 'fps', 'mm2px', 'food_radius']:
			return Characteristics(pd.DataFrame(
				[[key, chara_dict[key][name.name]] for key in chara_dict.keys()],
				columns=['id_exp', 'name']), name)
		elif name.name in ['food_center', 'traj_translation']:
			return Characteristics(pd.DataFrame(
				[[key]+chara_dict[key][name.name] for key in chara_dict.keys()],
				columns=['id_exp', 'x', 'y']), name)
		elif name.name in ['crop_limit_x', 'crop_limit_y']:
			return Characteristics(pd.DataFrame(
				[[key]+chara_dict[key][name.name] for key in chara_dict.keys()],
				columns=['id_exp', 'l_min', 'l_max']), name)
		elif name.name in ['ref_pts']:
			return Characteristics(pd.DataFrame(
				[[key]+chara_dict[key][name.name][0]+chara_dict[key][name.name][1] for key in chara_dict.keys()],
				columns=['id_exp', 'x1', 'y1', 'x2', 'y2']), name)
		elif name.name in ['entrance']:
			return Characteristics(pd.DataFrame(
				[[key]+chara_dict[key][name.name][0]+chara_dict[key][name.name][1]+chara_dict[key][name.name][2]
					for key in chara_dict.keys()],
				columns=['id_exp', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3']), name)
		else:
			print(name+' is not a characteristic')
