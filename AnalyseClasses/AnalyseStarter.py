import os

import pandas as pd

from Tools.JsonFiles import write_obj, import_obj


class AnalyseStarter:
    def __init__(self, root, group):
        self.root = root + group + '/'
        self.group = group

    def start(self, redo):
        self.__fill_and_write_definition_dict(redo)
        self.__sort_and_rewrite_markings()

    def __sort_and_rewrite_markings(self):
        add = self.root + 'Raw/markings.csv'
        df = pd.read_csv(add, index_col=['id_exp', 'id_ant', 'frame'])
        df.sort_index().to_csv(add)

    def __fill_and_write_definition_dict(self, redo):
        definition_dict = self.__init_definition_dict(redo)

        self.__fill_details_for_blob_features_and_markings(definition_dict)
        self.__fill_details_for_xy(definition_dict)
        self.__fill_details_for_absolute_orientation(definition_dict)
        self.__fill_details_for_experiment_features(definition_dict)
        self.__fill_details_for_entrance(definition_dict)
        self.__fill_details_for_ref_pts(definition_dict)

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
    def __fill_details_for_entrance(definition_dict):
        for i in [1, 2]:
            key = 'entrance' + str(i)
            definition_dict[key] = dict()
            definition_dict[key]['label'] = 'Entrance point ' + str(i)
            definition_dict[key]['category'] = 'Raw'
            definition_dict[key]['object_type'] = 'Characteristics2d'
            definition_dict[key]['description'] = 'One of the two points delimiting the entrance'

    @staticmethod
    def __fill_details_for_ref_pts(definition_dict):
        for i in [1, 2]:
            key = 'ref_pts' + str(i)
            definition_dict[key] = dict()
            definition_dict[key]['label'] = key.capitalize()
            definition_dict[key]['category'] = 'Raw'
            definition_dict[key]['object_type'] = 'Characteristics2d'
            definition_dict[key]['description'] = 'One of the two reference points'
            definition_dict[key]['label'] = 'Reference points ' + str(i)

    @staticmethod
    def __correct_object_types_for_som_experiment_features(definition_dict):
        definition_dict['food_center']['object_type'] = 'Characteristics2d'
        definition_dict['traj_translation']['object_type'] = 'Characteristics2d'
        definition_dict['crop_limit_x']['object_type'] = 'Characteristics2d'
        definition_dict['crop_limit_y']['object_type'] = 'Characteristics2d'

    @staticmethod
    def __correct_labels_for_some_experiment_features(definition_dict):
        definition_dict['n_frames']['label'] = 'Frame number'
        definition_dict['mm2px']['label'] = 'mm to px'
        definition_dict['food_radius']['label'] = 'Food radius'
        definition_dict['food_center']['label'] = 'Food center'
        definition_dict['traj_translation']['label'] = 'Trajectory translation'
        definition_dict['crop_limit_x']['label'] = 'Crop limit x'
        definition_dict['crop_limit_y']['label'] = 'Crop limit y'
        definition_dict['setup_orientation']['label'] = 'Setup orientation'

    @staticmethod
    def __fill_description_for_experiment_features(definition_dict):
        definition_dict['session']['description'] = 'trial number of the experiment'
        definition_dict['trial']['description'] = 'session number of the experiment'
        definition_dict['n_frames']['description'] = 'number of frames of the experiment'
        definition_dict['fps']['description'] = 'frame per second of the movie'
        definition_dict['mm2px']['description'] = 'ratio to convert millimeters to pixels'
        definition_dict['food_radius']['description'] = 'radius of the food piece'
        definition_dict['food_center']['description'] = 'coordinates of the center of the food piece'
        definition_dict['traj_translation'][
            'description'] = 'Translation vector between the picture and the cropped picture'
        definition_dict['crop_limit_x']['description'] = 'limits of the crop on the x coordinates'
        definition_dict['crop_limit_y']['description'] = 'limits of the crop on the y coordinates'
        definition_dict['setup_orientation']['description'] = 'Setup orientation'

    def __fill_details_for_experiment_features(self, definition_dict):
        self.__automatic_filling_details_for_experiment_features(definition_dict)
        self.__fill_description_for_experiment_features(definition_dict)
        self.__correct_labels_for_some_experiment_features(definition_dict)
        self.__correct_object_types_for_som_experiment_features(definition_dict)

    @staticmethod
    def __automatic_filling_details_for_experiment_features(definition_dict):
        for key in [
            'session', 'trial', 'n_frames', 'fps', 'mm2px', 'food_radius', 'setup_orientation',
            'food_center', 'traj_translation', 'crop_limit_x', 'crop_limit_y'
        ]:
            definition_dict[key] = dict()
            definition_dict[key]['label'] = key.capitalize()
            definition_dict[key]['category'] = 'Raw'
            definition_dict[key]['object_type'] = 'Characteristics1d'

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

    def __fill_details_for_blob_features_and_markings(self, definition_dict):
        self.__automatic_filling_for_blob_features_markings(definition_dict)
        self.__fill_description_for_blob_features(definition_dict)
        self.__correct_some_details_for_some_blob_features_and_markings(definition_dict)

    @staticmethod
    def __correct_some_details_for_some_blob_features_and_markings(definition_dict):
        definition_dict['major_axis_length']['label'] = 'major axis length'
        definition_dict['minor_axis_length']['label'] = 'minor axis length'
        definition_dict['markings']['description'] = 'Marking events'
        definition_dict['markings']['object_type'] = 'Events1d'

    @staticmethod
    def __fill_description_for_blob_features(definition_dict):
        definition_dict['area']['description'] = 'area'
        definition_dict['eccentricity']['description'] = 'eccentricity'
        definition_dict['major_axis_length']['description'] = 'major axis length'
        definition_dict['minor_axis_length']['description'] = 'minor axis length'
        definition_dict['perimeter']['description'] = 'perimeter'

    @staticmethod
    def __automatic_filling_for_blob_features_markings(definition_dict):
        for key in [
            'area', 'eccentricity',
            'major_axis_length', 'minor_axis_length', 'perimeter', 'markings'
        ]:
            definition_dict[key] = dict()
            definition_dict[key]['label'] = key.capitalize()
            definition_dict[key]['category'] = 'Raw'
            definition_dict[key]['object_type'] = 'TimeSeries1d'
