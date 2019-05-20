import os

import pandas as pd
import numpy as np

from cv2 import cv2
from matplotlib.path import Path

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Scripts.root import root
from Tools.MiscellaneousTools.Geometry import distance
from Tools.MiscellaneousTools.PickleJsonFiles import import_obj_json, write_obj_json


class AnalyseStarter:
    def __init__(self, root0, group, init_blobs=True):
        self.root = root0 + group + '/'
        self.group = group
        self.init_blobs = init_blobs
        self.characteristics = import_obj_json(self.root + 'Raw/Characteristics.json')

    def start(self, redo, markings=True, dynamic_food=False):
        self.__fill_and_write_definition_dict(redo, markings=markings, dynamic_food=dynamic_food)
        self.__sort_and_rewrite_trajectory()
        self.__sort_and_rewrite_markings(markings)
        self.__sort_and_rewrite_food(dynamic_food)

    def __sort_and_rewrite_markings(self, markings):
        if markings is True:
            print('write markings')
            add = self.root + 'Raw/markings.csv'
            df = pd.read_csv(add, index_col=[id_exp_name, id_ant_name, id_frame_name])
            df.sort_index().to_csv(add)

    def __sort_and_rewrite_trajectory(self):
        print('write trajectory')
        add = self.root + 'Raw/TimeSeries.csv'
        names = ['x0', 'y0', 'absoluteOrientation',
                 'area', 'eccentricity', 'major_axis_length', 'minor_axis_length', 'perimeter']

        exps = ExperimentGroupBuilder(root).build(self.group)
        exps.load(names[0])

        self.new_indexes = []
        exps.groupby(names[0], [id_exp_name, id_ant_name], self.__complete_time_series1d)
        new_indexes = pd.MultiIndex.from_tuples(self.new_indexes, names=[id_exp_name, id_ant_name, id_frame_name])

        res_df = exps.data_manager.data_loader.timeseries1d_loader.categories['Raw']
        res_df = res_df.reindex(new_indexes)
        res_df.to_csv(add)

    def __sort_and_rewrite_food(self, dynamic_food):
        if dynamic_food is True:
            print('write dynamic food')
            name_food_x = 'food_x0'
            name_food_y = 'food_y0'

            exps = ExperimentGroupBuilder(root).build(self.group)
            exps.load([name_food_x, name_food_y])

            res_df = exps.get_df(name_food_x)
            res_df = res_df.join(exps.get_df(name_food_y))
            res_df.sort_index(inplace=True)

            self.new_indexes = []
            exps.groupby(name_food_x, id_exp_name, self.__complete_chara_time_series1d)
            new_indexes = pd.MultiIndex.from_tuples(self.new_indexes, names=[id_exp_name, id_frame_name])

            res_df = res_df.reindex(new_indexes)

            add = self.root + 'Raw/CharacteristicTimeSeries.csv'
            res_df.to_csv(add)

    def __complete_time_series1d(self, df: pd.DataFrame):
        id_exp = df.index.get_level_values(id_exp_name)[0]
        id_ant = df.index.get_level_values(id_ant_name)[0]
        frame0, frame1 = df.index.get_level_values(id_frame_name)[[0, -1]]

        frames = range(frame0, frame1 + 1)
        exps = np.full(len(frames), id_exp)
        ants = np.full(len(frames), id_ant)
        self.new_indexes += list(zip(exps, ants, frames))

        return df

    def __complete_chara_time_series1d(self, df: pd.DataFrame):
        id_exp = df.index.get_level_values(id_exp_name)[0]
        frame0, frame1 = df.index.get_level_values(id_frame_name)[[0, -1]]

        frames = range(frame0, frame1 + 1)
        exps = np.full(len(frames), id_exp)
        self.new_indexes += list(zip(exps, frames))

        return df

    def __fill_and_write_definition_dict(self, redo, markings=True, dynamic_food=False):

        print('fill definition')

        definition_dict = self.__init_definition_dict(redo)

        self.__fill_details_for_basic_experiment_features(definition_dict)

        if self.init_blobs is True:
            self.__fill_details_for_blob_features(definition_dict)
            self.__fill_details_for_xy(definition_dict)
            self.__fill_details_for_absolute_orientation(definition_dict)

        self.__fill_details_for_markings(definition_dict, markings)
        self.__fill_details_for_dynamic_food(definition_dict, dynamic_food)

        self.__fill_details_food_radius_features(definition_dict)
        self.__fill_details_food_center_features(definition_dict)
        self.__fill_details_for_entrance(definition_dict)
        self.__fill_details_for_obstacle(definition_dict)
        self.__fill_details_setup_orientation(definition_dict)
        self.__fill_details_temporary_result(definition_dict)

        self.__write_definition_dict(definition_dict)

    def __init_definition_dict(self, redo):
        address = self.root + 'definition_dict.json'
        if redo is True or not (os.path.exists(address)) is True:
            def_dict = dict()
        else:
            def_dict = import_obj_json(address)
        return def_dict

    def __write_definition_dict(self, definition_dict):
        write_obj_json(self.root + 'definition_dict.json', definition_dict)

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
            definition_dict[key]['label'] = 'Food radius (mm)'
            definition_dict[key]['category'] = 'Raw'
            definition_dict[key]['object_type'] = 'Characteristics1d'
            definition_dict[key]['description'] = 'radius of the food (mm)'

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
    def __fill_details_for_markings(definition_dict, markings):
        if markings is True:
            key = 'markings'
            definition_dict[key] = dict()
            definition_dict[key]['label'] = key.capitalize()
            definition_dict[key]['object_type'] = 'Events1d'
            definition_dict[key]['category'] = 'Raw'
            definition_dict[key]['description'] = 'Marking events'

    @staticmethod
    def __fill_details_for_dynamic_food(definition_dict, dynamic_food):
        if dynamic_food is True:
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

    def compute_mm2px(self):

        mm2px = 'mm2px'
        exps = ExperimentGroupBuilder(root).build(self.group)
        exps.load(mm2px)

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 2
        detector = cv2.SimpleBlobDetector_create(params)
        for id_exp in exps.get_index(mm2px):

            bg_img = exps.get_bg_img(id_exp)
            bg_img = np.sqrt(bg_img[400:600, 900:1100]).astype(np.uint8)
            ret, bg_img = cv2.threshold(bg_img, 9.5, 255, cv2.THRESH_BINARY)

            points = detector.detect(bg_img)
            nbr_pts = len(points)

            ds = np.zeros((nbr_pts, nbr_pts))
            for i in range(nbr_pts):
                for j in range(i + 1, nbr_pts):
                    ds[i, j] = distance(points[i].pt, points[j].pt)

            h = np.histogram(ds[ds != 0])

            n_max = 4
            mask = np.where(h[0] > n_max)[0]
            while len(mask) == 0:
                n_max -= 1
                mask = np.where(h[0] > n_max)[0]

            d0 = h[1][mask[0]]
            d1 = h[1][mask[0] + 1]
            d = np.max(ds[(ds >= d0) * (ds <= d1)])

            exps.get_df(mm2px).loc[id_exp] = round(d/21., 3)

        exps.write(mm2px)

    def compute_exit0(self):
        result_name1 = 'exit0_1'
        result_name2 = 'exit0_2'

        exps = ExperimentGroupBuilder(root).build(self.group)
        exps.load(['food_x0', 'food_y0', 'entrance1', 'entrance2', 'traj_translation'])

        exps.add_copy(old_name='entrance1', new_name=result_name1,
                      category='Raw', label='Exit position 1 (setup system)',
                      xlabel='x coordinates', ylabel='y coordinates',
                      description='Coordinates of one of the points defining the exit in the setup system')

        exps.add_copy(old_name='entrance2', new_name=result_name2,
                      category='Raw', label='Exit position 2 (setup system)',
                      xlabel='x coordinates', ylabel='y coordinates',
                      description='Coordinates of one of the points defining the exit in the setup system')

        exps.operation_between_2names(result_name1, 'traj_translation', lambda x, y: x-y, 'x', 'x')
        exps.operation_between_2names(result_name1, 'traj_translation', lambda x, y: x-y, 'y', 'y')

        exps.operation_between_2names(result_name2, 'traj_translation', lambda x, y: x-y, 'x', 'x')
        exps.operation_between_2names(result_name2, 'traj_translation', lambda x, y: x-y, 'y', 'y')

        exps.write([result_name1, result_name2])

    def compute_is_from_outside(self):
        result_name = 'from_outside'
        category = 'CleanedRaw'

        exps = ExperimentGroupBuilder(root).build(self.group)
        if exps.is_name_existing('decrossed_x0'):
            exps.load_as_2d('decrossed_x0', 'decrossed_y0', 'xy')
        else:
            exps.load_as_2d('x0', 'y0', 'xy', 'x', 'y')

        self.__compute_gate_pts(exps)

        def is_from_outside4each_group(df: pd.DataFrame):
            id_exp = int(df.index.get_level_values('id_exp')[0])
            gate_path = Path([exps.gate1.df.loc[id_exp], exps.gate2.df.loc[id_exp],
                              exps.gate3.df.loc[id_exp], exps.gate4.df.loc[id_exp]])

            xys = np.array(df)[:10]
            df[:] = np.nan
            from_outside = any(gate_path.contains_points(xys))
            df.iloc[0, :] = from_outside

            return df

        exps.xy.df = exps.xy.df.groupby(['id_exp', 'id_ant']).apply(is_from_outside4each_group)
        df_res = exps.xy.df.dropna()
        df_res.index = df_res.index.droplevel('frame')
        df_res = df_res.drop(columns='y')

        exps.add_new1d_from_df(df=df_res.astype(int), name=result_name, object_type='AntCharacteristics1d',
                               category=category, label='Is the ant from outside?',
                               description='Boolean saying if the ant is coming from outside or not')

        exps.write(result_name)

    @staticmethod
    def __compute_gate_pts(exps):
        exps.load(['entrance1', 'entrance2', 'mm2px', 'traj_translation'])

        for gate_num in range(1, 5):
            exps.add_copy('entrance1', 'gate'+str(gate_num))

        for id_exp in exps.id_exp_list:

            xmin = min(exps.entrance1.df.loc[id_exp].x, exps.entrance2.df.loc[id_exp].x)
            ymin = min(exps.entrance1.df.loc[id_exp].y, exps.entrance2.df.loc[id_exp].y)
            ymax = max(exps.entrance1.df.loc[id_exp].y, exps.entrance2.df.loc[id_exp].y)
            xmax = max(exps.entrance1.df.loc[id_exp].x, exps.entrance2.df.loc[id_exp].x)

            dl = 20*exps.get_value('mm2px', id_exp)
            exps.gate1.df.x.loc[id_exp] = xmin - dl
            exps.gate1.df.y.loc[id_exp] = ymin - dl
            exps.gate2.df.x.loc[id_exp] = xmin - dl
            exps.gate2.df.y.loc[id_exp] = ymax + dl
            exps.gate3.df.x.loc[id_exp] = xmax + dl
            exps.gate3.df.y.loc[id_exp] = ymax + dl
            exps.gate4.df.x.loc[id_exp] = xmax + dl
            exps.gate4.df.y.loc[id_exp] = ymin - dl

        for gate_num in range(1, 5):
            exps.operation_between_2names('gate'+str(gate_num), 'traj_translation', lambda x, y: x - y, 'x', 'x')
            exps.operation_between_2names('gate'+str(gate_num), 'traj_translation', lambda x, y: x - y, 'y', 'y')
