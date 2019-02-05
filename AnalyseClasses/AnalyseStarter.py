import os

import pandas as pd

from Tools.MiscellaneousTools.JsonFiles import import_obj, write_obj


class AnalyseStarter:
    def __init__(self, root, group, init_blobs=True):
        self.root = root + group + '/'
        self.group = group
        self.init_blobs = init_blobs
        self.characteristics = import_obj(self.root + '/Raw/Characteristics.json')

    def start(self, redo, markings=True, dynamic_food=False):
        self.__fill_and_write_definition_dict(redo, markings=markings, dynamic_food=dynamic_food)
        if markings is True:
            self.__sort_and_rewrite_markings()
        if dynamic_food is True:
            self.__sort_and_rewrite_food()

    def __sort_and_rewrite_markings(self):
        add = self.root + 'Raw/markings.csv'
        df = pd.read_csv(add, index_col=['id_exp', 'id_ant', 'frame'])
        df.sort_index().to_csv(add)

    def __sort_and_rewrite_food(self):
        add = self.root + 'Raw/CharacteristicTimeSeries.csv'
        df = pd.read_csv(add, index_col=['id_exp', 'frame', 'food_x0', 'food_y0'])
        df.sort_index().to_csv(add)

    def __fill_and_write_definition_dict(self, redo, markings=True, dynamic_food=False):

        definition_dict = self.__init_definition_dict(redo)

        self.__fill_details_for_basic_experiment_features(definition_dict)

        if self.init_blobs:
            self.__fill_details_for_blob_features(definition_dict)
            self.__fill_details_for_xy(definition_dict)
            self.__fill_details_for_absolute_orientation(definition_dict)

        if markings is True:
            self.__fill_details_for_markings(definition_dict)
        if dynamic_food is True:
            self.__fill_details_for_dynamic_food(definition_dict)

        self.__fill_details_food_radius_features(definition_dict)
        self.__fill_details_food_center_features(definition_dict)
        self.__fill_details_for_entrance(definition_dict)
        self.__fill_details_for_obstacle(definition_dict)
        self.__fill_details_setup_orientation(definition_dict)
        self.__fill_details_temporary_result(definition_dict)

        self.__write_definition_dict(definition_dict)

    def __init_definition_dict(self, redo):
        address = self.root + 'definition_dict.json'
        if redo or not (os.path.exists(address)):
            def_dict = dict()
        else:
            def_dict = import_obj(address)
        return def_dict

    def __write_definition_dict(self, definition_dict):
        write_obj(self.root + 'definition_dict.json', definition_dict)

    @staticmethod
    def __fill_details_for_blob_features(definition_dict):
        for key in [
            'area', 'eccentricity',
            'major_axis_length', 'minor_axis_length', 'perimeter'
        ]:
            definition_dict[key] = dict()
            definition_dict[key]['label'] = key.capitalize()
            definition_dict[key]['category'] = 'Raw'
            definition_dict[key]['object_type'] = 'TimeSeries1d'

        definition_dict['area']['description'] = 'area'
        definition_dict['eccentricity']['description'] = 'eccentricity'
        definition_dict['major_axis_length']['description'] = 'major axis length'
        definition_dict['minor_axis_length']['description'] = 'minor axis length'
        definition_dict['perimeter']['description'] = 'perimeter'

        definition_dict['major_axis_length']['label'] = 'major axis length'
        definition_dict['minor_axis_length']['label'] = 'minor axis length'

    @staticmethod
    def __fill_details_for_basic_experiment_features(definition_dict):
        for key in [
            'session', 'trial', 'date', 'temperature', 'humidity', 'n_frames', 'fps', 'mm2px',
            'food_center', 'traj_translation', 'crop_limit_x', 'crop_limit_y'
        ]:
            definition_dict[key] = dict()
            definition_dict[key]['label'] = key.capitalize()
            definition_dict[key]['category'] = 'Raw'
            definition_dict[key]['object_type'] = 'Characteristics1d'

        definition_dict['session']['description'] = 'trial number of the experiment'
        definition_dict['trial']['description'] = 'session number of the experiment'
        definition_dict['n_frames']['description'] = 'number of frames of the experiment'
        definition_dict['fps']['description'] = 'frame per second of the movie'
        definition_dict['mm2px']['description'] = 'ratio to convert millimeters to pixels'
        definition_dict['traj_translation'][
            'description'] = 'Translation vector between the picture and the cropped picture'
        definition_dict['crop_limit_x']['description'] = 'limits of the crop on the x coordinates'
        definition_dict['crop_limit_y']['description'] = 'limits of the crop on the y coordinates'
        definition_dict['date']['description'] = 'Date of experiment'
        definition_dict['temperature']['description'] = 'Air temperature during experiment'
        definition_dict['humidity']['description'] = 'Air humidity during experiment'

        definition_dict['n_frames']['label'] = 'Frame number'
        definition_dict['mm2px']['label'] = 'mm to px'
        definition_dict['traj_translation']['label'] = 'Trajectory translation'
        definition_dict['crop_limit_x']['label'] = 'Crop limit x'
        definition_dict['crop_limit_y']['label'] = 'Crop limit y'

        definition_dict['traj_translation']['object_type'] = 'Characteristics2d'
        definition_dict['crop_limit_x']['object_type'] = 'Characteristics2d'
        definition_dict['crop_limit_y']['object_type'] = 'Characteristics2d'

    def __fill_details_food_radius_features(self, definition_dict):
        key = 'food_radius'
        if key in self.characteristics[list(self.characteristics.keys())[0]]:
            definition_dict[key] = dict()
            definition_dict[key]['label'] = 'Food radius'
            definition_dict[key]['category'] = 'Raw'
            definition_dict[key]['object_type'] = 'Characteristics1d'
            definition_dict[key]['description'] = 'radius of the food (px)'

    def __fill_details_food_center_features(self, definition_dict):
        key = 'food_center'
        if key in self.characteristics[list(self.characteristics.keys())[0]]:
            definition_dict[key] = dict()
            definition_dict[key]['label'] = 'Food center'
            definition_dict[key]['category'] = 'Raw'
            definition_dict[key]['object_type'] = 'Characteristics2d'
            definition_dict[key]['description'] = 'coordinates of the center of the food'

    @staticmethod
    def __fill_details_for_entrance(definition_dict):
        for i in [1, 2]:
            key = 'entrance' + str(i)
            definition_dict[key] = dict()
            definition_dict[key]['label'] = 'Entrance point ' + str(i)
            definition_dict[key]['category'] = 'Raw'
            definition_dict[key]['object_type'] = 'Characteristics2d'
            definition_dict[key]['description'] = 'One of the two points delimiting the entrance'

    def __fill_details_setup_orientation(self, definition_dict):
        key = 'setup_orientation'
        if key in self.characteristics[list(self.characteristics.keys())[0]]:
            definition_dict[key] = dict()
            definition_dict[key]['label'] = 'Setup orientation'
            definition_dict[key]['category'] = 'Raw'
            definition_dict[key]['object_type'] = 'Characteristics1d'
            definition_dict[key]['description'] = 'orientation of the Setup'

    def __fill_details_temporary_result(self, definition_dict):
        key = 'temporary_result'
        if key in self.characteristics[list(self.characteristics.keys())[0]]:
            definition_dict[key] = dict()
            definition_dict[key]['label'] = 'Preliminary results'
            definition_dict[key]['category'] = 'Raw'
            definition_dict[key]['object_type'] = 'Characteristics1d'
            definition_dict[key]['description'] = 'Preliminary results got by watching movies'

    def __fill_details_for_obstacle(self, definition_dict):
        if 'obstacle1' in self.characteristics[list(self.characteristics.keys())[0]]:
            for i in [1, 2]:
                key = 'obstacle' + str(i)
                definition_dict[key] = dict()
                definition_dict[key]['label'] = 'Obstacle point ' + str(i)
                definition_dict[key]['category'] = 'Raw'
                definition_dict[key]['object_type'] = 'Characteristics2d'
                definition_dict[key]['description'] = 'Point at one of the two ends of the obstacle'

    @staticmethod
    def __fill_details_for_absolute_orientation(definition_dict):
        key = 'absoluteOrientation'
        definition_dict[key] = dict()
        definition_dict[key]['label'] = 'Absolute orientation'
        definition_dict[key]['category'] = 'Raw'
        definition_dict[key]['object_type'] = 'TimeSeries1d'
        definition_dict[key]['description'] = 'absolute orientation'

    @staticmethod
    def __fill_details_for_xy(definition_dict):
        for key in ['x', 'y']:
            key2 = key + '0'
            definition_dict[key2] = dict()
            definition_dict[key2]['label'] = key2
            definition_dict[key2]['category'] = 'Raw'
            definition_dict[key2]['object_type'] = 'TimeSeries1d'
            definition_dict[key2]['description'] = key + ' coordinate (in the cropped image system)'

    @staticmethod
    def __fill_details_for_markings(definition_dict):
        key = 'markings'
        definition_dict[key] = dict()
        definition_dict[key]['label'] = key.capitalize()
        definition_dict[key]['object_type'] = 'Events1d'
        definition_dict[key]['category'] = 'Raw'
        definition_dict[key]['description'] = 'Marking events'

    @staticmethod
    def __fill_details_for_dynamic_food(definition_dict):
        key = 'food_x0'
        definition_dict[key] = dict()
        definition_dict[key]['label'] = 'X0 food'
        definition_dict[key]['category'] = 'Raw'
        definition_dict[key]['object_type'] = 'CharacteristicTimeSeries1d'
        definition_dict[key]['description'] = 'X coordinates of the food trajectory'

        key = 'food_y0'
        definition_dict[key] = dict()
        definition_dict[key]['label'] = 'Y0 food'
        definition_dict[key]['category'] = 'Raw'
        definition_dict[key]['object_type'] = 'CharacteristicTimeSeries1d'
        definition_dict[key]['description'] = 'Y coordinates of the food trajectory'
