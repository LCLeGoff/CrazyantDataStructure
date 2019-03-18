import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from cv2 import cv2

from DataStructure.DataManager.DataFileManager import DataFileManager
from DataStructure.Builders.Builder import Builder
from DataStructure.DataObjects.CharacteristicTimeSeries2d import CharacteristicTimeSeries2dBuilder
from DataStructure.DataObjects.Events2d import Events2dBuilder
from DataStructure.DataObjects.Filters import Filters
from DataStructure.DataObjects.TimeSeries2d import TimeSeries2dBuilder
from DataStructure.VariableNames import dataset_name, id_exp_name, id_ant_name, id_frame_name
from Movies.Movies import Movies
from Scripts.root import root_movie
from Tools.MiscellaneousTools.ArrayManipulation import running_mean
from Tools.MiscellaneousTools.Geometry import norm_angle_tab
from Tools.MiscellaneousTools.PickleJsonFiles import import_obj_pickle
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager
from Tools.Plotter.Plotter import Plotter


class ExperimentGroups:

    def __init__(self, root, group):
        self.root = root + group + '/'
        self.group = group
        self.data_manager = DataFileManager(root, group)
        self.pandas_index_manager = PandasIndexManager()

        self.timeseries_exp_ant_frame_index = None
        self.timeseries_exp_frame_ant_index = None
        self.timeseries_exp_ant_index = import_obj_pickle(self.root + 'TimeSeries_exp_ant_index.p')
        self.timeseries_exp_frame_index = import_obj_pickle(self.root + 'TimeSeries_exp_frame_index.p')
        self.characteristic_timeseries_exp_frame_index = import_obj_pickle(
            self.root + 'CharacteristicTimeSeries_exp_frame_index.p')
        self.id_exp_list = list(self.timeseries_exp_frame_index.keys())
        self.id_exp_list.sort()

        self.ref_id_exp = np.array(pd.read_csv(self.root + 'ref_id_exp.csv'))

        self.names = set()

    def load_timeseries_exp_ant_frame_index(self):
        if self.timeseries_exp_ant_frame_index is None:
            self.timeseries_exp_ant_frame_index = import_obj_pickle(self.root + 'TimeSeries_exp_ant_frame_index.p')

    def load_timeseries_exp_frame_ant_index(self):
        if self.timeseries_exp_frame_ant_index is None:
            self.timeseries_exp_frame_ant_index = import_obj_pickle(self.root + 'TimeSeries_exp_frame_ant_index.p')

    @staticmethod
    def turn_to_list(names):
        if isinstance(names, str):
            names = [names]
        return names

    def get_data_object(self, name):
        return self.__dict__[name]

    def get_columns(self, name):
        return self.get_data_object(name).df.columns

    def get_xname(self, name):
        return self.get_data_object(name).df.columns[0]

    def get_yname(self, name):
        return self.get_data_object(name).df.columns[1]

    def get_category(self, name):
        return self.get_data_object(name).definition.category

    def get_object_type(self, name):
        return self.get_data_object(name).object_type

    def get_object_type_in_1d(self, name):
        object_type = self.get_data_object(name).object_type
        object_type = object_type[:-2] + '1d'
        return object_type

    def get_label(self, name):
        return self.get_data_object(name).definition.label

    def get_xlabel(self, name):
        return self.get_data_object(name).definition.xlabel

    def get_ylabel(self, name):
        return self.get_data_object(name).definition.ylabel

    def get_description(self, name):
        return self.get_data_object(name).definition.description

    def get_df(self, name):
        return self.get_data_object(name).df

    def get_index(self, name):
        return self.get_data_object(name).df.index

    def get_index_names(self, name):
        return list(self.get_df(name).index.names)

    def get_ref_id_exp(self, id_exp):
        return tuple(self.ref_id_exp[self.ref_id_exp[:, 0] == id_exp, 1:][0, :])

    def get_movie_address(self, id_exp):
        session, trial = self.get_ref_id_exp(id_exp)
        return root_movie + self.group + '/' + self.group + '_S' + str(session).zfill(2) + '_T' + str(trial).zfill(
            2) + '.MP4'

    def get_movie(self, id_exp):
        address = self.get_movie_address(id_exp)
        return Movies(address, id_exp)

    def get_bg_img_address(self, id_exp):
        return self.root + '/I_bg/' + str(id_exp).zfill(3) + '.png'

    def get_bg_img(self, id_exp):
        address = self.get_bg_img_address(id_exp)
        return cv2.imread(address, cv2.IMREAD_GRAYSCALE)

    def get_array_all_indexes(self, name):
        return self.pandas_index_manager.get_unique_index_array(df=self.get_df(name))

    def get_array_id_exp_ant(self, name):
        return self.pandas_index_manager.get_unique_index_array(df=self.get_df(name),
                                                                index_names=[id_exp_name, id_ant_name])

    def get_dict_id_exp_ant(self, name):
        if self.get_object_type(name) in ['TimeSeries1d', 'TimeSeries2d']:
            return self.timeseries_exp_ant_index
        else:
            return self.pandas_index_manager.get_index_dict(df=self.get_df(name),
                                                            index_names=[id_exp_name, id_ant_name])

    def get_dict_id_exp_ant_frame(self, name):
        if self.get_object_type(name) in ['TimeSeries1d', 'TimeSeries2d']:
            return self.timeseries_exp_ant_frame_index
        else:
            return self.pandas_index_manager.get_index_dict(
                df=self.get_df(name), index_names=[id_exp_name, id_ant_name, id_frame_name])

    def plot_traj_on_movie(self, traj_names, id_exp, frame, id_ants=None):
        self.load_timeseries_exp_frame_ant_index()
        if id_ants is None:
            id_ants = self.timeseries_exp_frame_ant_index[id_exp][frame]
        elif len(np.array(id_ants)) == 1:
            id_ants = [id_ants]

        self.load(traj_names)

        mov = self.get_movie(id_exp)
        img_frame = mov.get_frame(frame)

        # plt.figure(figsize=(12, 12))
        plt.imshow(img_frame, cmap='gray')
        for id_ant in id_ants:

            if (id_exp, id_ant, frame) in self.__dict__[traj_names[0]].df.index:
                x, y = np.array(self.__dict__[traj_names[0]].df.loc[id_exp, id_ant, frame])
                orientation = np.array(self.__dict__[traj_names[1]].df.loc[id_exp, id_ant, frame])
                x, y = self.convert_xy_to_movie_system(id_exp, x, y)
                orientation = self.convert_orientation_to_movie_system(id_exp, orientation)

                lg = 10
                plt.plot(x, y, '.')
                plt.plot(x + np.array([-1, 1]) * lg * np.cos(orientation) / 2.,
                         y + np.array([-1, 1]) * lg * np.sin(orientation) / 2.)

        plt.title(frame)
        plt.show()

    def set_id_exp_list(self, id_exp_list=None):
        if id_exp_list is None:
            id_exp_list = self.id_exp_list
        return id_exp_list

    def add_object(self, name, obj, replace=False):
        if not replace and name in self.names:
            raise NameError(name + ' exists already')
        else:
            self.__dict__[name] = obj
            self.names.add(name)

    def remove_object(self, names):
        names = self.turn_to_list(names)
        for name in names:
            self.__dict__.pop(name)
            self.names.remove(name)

    def load(self, names):
        names = self.turn_to_list(names)
        for name in names:
            if name not in self.__dict__.keys():
                print('loading ', name)
                self.add_object(name, self.data_manager.load(name), replace=True)

    def write(self, names):
        names = self.turn_to_list(names)
        for name in names:
            print('writing ', name)
            self.data_manager.write(self.get_data_object(name))

    def delete_data(self, names):
        names = self.turn_to_list(names)
        self.load(names)
        for name in names:
            print('deleting ', name)
            self.data_manager.delete(self.get_data_object(name))

    def is_name_in_data(self, name):
        return self.data_manager.is_name_in_data(name)

    def load_as_2d(
            self, name1, name2, result_name,
            xname=None, yname=None,
            category=None, label=None, xlabel=None, ylabel=None, description=None):

        self.load([name1, name2])

        is_name1_time_series_1d = self.get_object_type(name1) == 'TimeSeries1d'
        is_name2_time_series_1d = self.get_object_type(name2) == 'TimeSeries1d'

        if is_name1_time_series_1d and is_name2_time_series_1d:
            self.add_2d_from_1ds(
                name1=name1, name2=name2, result_name=result_name,
                xname=xname, yname=yname, category=category,
                label=label, xlabel=xlabel, ylabel=ylabel, description=description)

    def __is_1d(self, name):
        object_type = self.get_object_type(name)
        if object_type in ['Events2d', 'TimeSeries2d', 'Characteristics2d', 'CharacteristicTimeSeries2d']:
            return False
        elif object_type in [
                'Events1d', 'TimeSeries1d', 'Characteristics1d', 'AntCharacteristics1d', 'CharacteristicTimeSeries1d']:
            return True
        elif object_type in [dataset_name]:
            return self.get_data_object(name).get_dimension() == 1
        else:
            raise TypeError('Object type ' + object_type + ' unknown')

    def __is_indexed_by_exp(self, name):
        return set(self.get_index_names(name)) == {id_exp_name}

    def __is_indexed_by_exp_ant_frame(self, name):
        return set(self.get_index_names(name)) == {id_exp_name, id_ant_name, id_frame_name}

    def __is_indexed_by_exp_ant(self, name):
        return set(self.get_index_names(name)) == {id_exp_name, id_ant_name}

    def __is_indexed_by_exp_frame(self, name):
        return set(self.get_index_names(name)) == {id_exp_name, id_frame_name}

    def __is_frame_in_indexes(self, name):
        return id_frame_name in self.get_index_names(name)

    def __is_a_time_series(self, name):
        object_type = self.get_object_type(name)
        return object_type in [
            'TimeSeries1d', 'TimeSeries2d',
            'CharacteristicTimeSeries1d', 'CharacteristicTimeSeries2d']

    def __is_same_index_names(self, name1, name2):
        index_name1 = self.get_index_names(name1)
        index_name2 = self.get_index_names(name2)
        return set(index_name1) == set(index_name2)

    def rename_data(
            self, old_name, new_name=None, xname=None, yname=None, category=None,
            label=None, xlabel=None, ylabel=None, description=None):
        print('renaming ', old_name, 'to', new_name)

        self.load(old_name)
        self.data_manager.rename(
            self.get_data_object(old_name), name=new_name, xname=xname, yname=yname, category=category,
            label=label, xlabel=xlabel, ylabel=ylabel, description=description)

    def rename(
            self, old_name, new_name=None, xname=None, yname=None, category=None,
            label=None, xlabel=None, ylabel=None, description=None):
        self.load(old_name)
        if self.__is_1d(old_name):
            self.rename1d(old_name=old_name, new_name=new_name, category=category, label=label, description=description)
        else:
            self.rename2d(
                old_name=old_name, new_name=new_name, xname=xname, yname=yname, category=category,
                label=label, xlabel=xlabel, ylabel=ylabel, description=description)
        if old_name == new_name:
            self.add_object(new_name, self.get_data_object(old_name))
            self.remove_object(old_name)

    def rename1d(self, old_name, new_name=None, category=None, label=None, description=None):
        if new_name is None:
            new_name = old_name
        self.__dict__[old_name].rename(name=new_name, category=category, label=label, description=description)

    def rename2d(
            self, old_name, new_name=None, xname=None, yname=None, category=None,
            label=None, xlabel=None, ylabel=None, description=None):

        if new_name is None:
            new_name = old_name
        self.__dict__[old_name].rename(
            name=new_name, xname=xname, yname=yname, category=category,
            label=label, xlabel=xlabel, ylabel=ylabel, description=description)

    def add_copy(
            self, old_name, new_name, copy_definition=False,
            category=None, label=None, xlabel=None, ylabel=None, description=None,
            replace=False):

        if self.__is_1d(old_name):
            self.add_copy1d(
                name_to_copy=old_name, copy_name=new_name, copy_definition=copy_definition,
                category=category, label=label, description=description, replace=replace)
        else:
            self.add_copy2d(
                name_to_copy=old_name, copy_name=new_name, copy_definition=copy_definition,
                category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description,
                replace=replace)

    def add_2d_from_1ds(
            self, name1, name2, result_name, xname=None, yname=None,
            category=None, label=None, xlabel=None, ylabel=None, description=None, replace=False):

        object_type1 = self.get_object_type(name1)
        object_type2 = self.get_object_type(name2)

        if object_type1 == object_type2 and self.get_data_object(name1).df.index.equals(
                self.get_data_object(name2).df.index):
            if xname is None:
                xname = name1
            if yname is None:
                yname = name2
            if object_type1 == 'Events1d':
                event = Events2dBuilder().build_from_1d(
                    event1=self.get_data_object(name1),
                    event2=self.get_data_object(name2), name=result_name, xname=xname,
                    yname=yname, category=category, label=label, xlabel=xlabel,
                    ylabel=ylabel, description=description)
                self.add_object(result_name, event, replace)
            elif object_type1 == 'TimeSeries1d':
                ts = TimeSeries2dBuilder().build_from_1d(
                    ts1=self.get_data_object(name1),
                    ts2=self.get_data_object(name2), name=result_name, xname=xname,
                    yname=yname, category=category, label=label, xlabel=xlabel,
                    ylabel=ylabel, description=description)
                self.add_object(result_name, ts, replace)
            elif object_type1 == 'CharacteristicTimeSeries1d':
                ts = CharacteristicTimeSeries2dBuilder().build_from_1d(
                    ts1=self.get_data_object(name1),
                    ts2=self.get_data_object(name2), name=result_name, xname=xname,
                    yname=yname, category=category, label=label, xlabel=xlabel,
                    ylabel=ylabel, description=description)
                self.add_object(result_name, ts, replace)
            else:
                raise TypeError(object_type1 + ' can not be gathered in 2d')

    def add_copy1d(
            self, name_to_copy, copy_name,
            category=None, label=None, description=None, copy_definition=False, replace=False):

        if copy_definition:
            obj = self.get_data_object(name_to_copy).copy(
                name=copy_name,
                category=self.get_category(name_to_copy),
                label=self.get_label(name_to_copy),
                description=self.get_description(name_to_copy))
        else:
            if category is None:
                category = self.get_category(name_to_copy)
            obj = self.get_data_object(name_to_copy).copy(
                name=copy_name, category=category, label=label,
                description=description)

        self.add_object(copy_name, obj, replace)

    def add_copy2d(
            self, name_to_copy, copy_name, new_xname=None, new_yname=None,
            category=None, label=None, xlabel=None, ylabel=None, description=None,
            copy_definition=False, replace=False):

        if copy_definition:
            obj = self.get_data_object(name_to_copy).copy(
                name=copy_name,
                xname=self.get_xname(name_to_copy),
                yname=self.get_yname(name_to_copy),
                category=self.get_category(name_to_copy),
                label=self.get_label(name_to_copy),
                xlabel=self.get_xlabel(name_to_copy),
                ylabel=self.get_ylabel(name_to_copy),
                description=self.get_description(name_to_copy)
            )
        else:
            if new_xname is None:
                new_xname = self.get_xname(name_to_copy)

            if new_yname is None:
                new_yname = self.get_yname(name_to_copy)

            obj = self.get_data_object(name_to_copy).copy(
                name=copy_name, xname=new_xname, yname=new_yname,
                category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description)

        self.add_object(copy_name, obj, replace)

    def add_new1d_empty(self, name, object_type, category=None, label=None, description=None, replace=False):
        df = self.__create_empty_df(name, object_type)
        obj = Builder.build1d_from_df(
            df=df, name=name, object_type=object_type, category=category, label=label, description=description)
        self.add_object(name, obj, replace=replace)

    def __create_empty_df(self, name, object_type, index_names=None):
        if object_type in ['TimeSeries1d', 'TimeSeries2d', 'Events1d', 'Events2d']:
            df = self.pandas_index_manager.create_empty_df(column_names=name,
                                                           index_names=[id_exp_name, id_ant_name, id_frame_name])
        elif object_type in ['AntCharacteristics1d']:
            df = self.pandas_index_manager.create_empty_df(column_names=name, index_names=[id_exp_name, id_ant_name])
        elif object_type in ['Characteristics1d', 'Characteristics2d']:
            df = self.pandas_index_manager.create_empty_df(column_names=name, index_names=id_exp_name)
        elif object_type in ['CharacteristicTimeSeries1d', 'CharacteristicTimeSeries2d']:
            df = self.pandas_index_manager.create_empty_df(column_names=name, index_names=[id_exp_name, id_frame_name])
        elif object_type in [dataset_name]:
            df = self.pandas_index_manager.create_empty_df(column_names=name, index_names=index_names)
        else:
            raise IndexError('Object type ' + object_type + ' unknown')
        return df

    def add_new2d_empty(
            self, name, xname, yname, object_type, category=None,
            label=None, xlabel=None, ylabel=None, description=None, replace=False):

        obj = Builder.build2d_from_array(
            array=np.zeros((0, 5)), name=name, xname=xname, yname=yname,
            object_type=object_type, category=category,
            label=label, xlabel=xlabel, ylabel=ylabel, description=description)

        self.add_object(name, obj, replace)

    def add_new1d_from_df(self, df, name, object_type, category=None, label=None, description=None, replace=False):
        obj = Builder.build1d_from_df(
            df=df, name=name, object_type=object_type, category=category, label=label, description=description)
        self.add_object(name, obj, replace)

    def add_new2d_from_df(
            self, df, name, xname, yname, object_type, category=None,
            label=None, xlabel=None, ylabel=None, description=None, replace=False):

        obj = Builder.build2d_from_df(
            df=df, name=name, xname=xname, yname=yname,
            object_type=object_type, category=category,
            label=label, xlabel=xlabel, ylabel=ylabel, description=description)

        self.add_object(name, obj, replace=replace)

    def add_new1d_from_array(
            self, array, name, object_type, category=None, label=None, description=None, replace=False):

        df = self.__convert_array_to_df(array, name, object_type)

        obj = Builder.build1d_from_df(
            df=df, name=name, object_type=object_type, category=category, label=label, description=description)

        self.add_object(name, obj, replace=replace)

    def __convert_array_to_df(self, array, name, object_type, index_names=None):

        if object_type in ['TimeSeries1d', 'TimeSeries2d', 'Events1d', 'Events2d']:
            df = self.pandas_index_manager.convert_array_to_df(
                array, index_names=[id_exp_name, id_ant_name, id_frame_name], column_names=name)

        elif object_type in ['AntCharacteristics1d', 'AntCharacteristics2d']:
            df = self.pandas_index_manager.convert_array_to_df(
                array, index_names=[id_exp_name, id_ant_name], column_names=name)

        elif object_type in ['Characteristics1d', 'Characteristics2d']:
            df = self.pandas_index_manager.convert_array_to_df(array, index_names=id_exp_name, column_names=name)

        elif object_type in ['CharacteristicTimeSeries1d', 'CharacteristicTimeSeries2d']:
            df = self.pandas_index_manager.convert_array_to_df(
                array, index_names=[id_exp_name, id_frame_name], column_names=name)

        elif object_type in [dataset_name]:
            df = self.pandas_index_manager.convert_array_to_df(array, index_names=index_names, column_names=name)

        else:
            raise IndexError('Object type ' + object_type + ' unknown')

        return df

    def get_reindexed_df(self, name_to_reindex, reindexer_name, fill_value=None):

        if self.__is_indexed_by_exp_ant_frame(reindexer_name) is True:

            if self.__is_indexed_by_exp_ant_frame(reindexer_name) is True:

                df = self.get_df(name_to_reindex).reindex(self.get_index(reindexer_name), fill_value=fill_value)
                return df
            else:
                id_exps = self.get_index(reindexer_name).get_level_values(id_exp_name)
                id_ants = self.get_index(reindexer_name).get_level_values(id_ant_name)
                frames = self.get_index(reindexer_name).get_level_values(id_frame_name)

                if self.__is_indexed_by_exp(name_to_reindex) is True:

                    df = self.get_df(name_to_reindex).reindex(id_exps, fill_value=fill_value)
                    df[id_ant_name] = id_ants
                    df[id_frame_name] = frames
                    df.reset_index(inplace=True)
                    df.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)
                    return df

                elif self.__is_indexed_by_exp_frame(reindexer_name) is True:

                    idxs = pd.MultiIndex.from_tuples(list(zip(id_exps, frames)), names=[id_exp_name, id_frame_name])

                    df = self.get_df(name_to_reindex).reindex(idxs, fill_value=fill_value)
                    df[id_ant_name] = id_ants
                    df.reset_index(inplace=True)
                    df.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)
                    return df

                else:
                    raise TypeError(
                        'reindexing a ' + str(list(self.get_index(name_to_reindex).get_names))
                        + ' indexed object by a ' + str(list(self.get_index(reindexer_name).get_names))
                        + ' indexed object is not implemented yet')
        else:
            raise TypeError(
                'reindexing a ' + str(list(self.get_index(name_to_reindex).get_names))
                + ' indexed object by a ' + str(list(self.get_index(reindexer_name).get_names))
                + ' indexed object is not implemented yet')

    def filter_with_values(
            self, name_to_filter, filter_name, filter_values=1, result_name=None,
            xname=None, yname=None, category=None, label=None, xlabel=None, ylabel=None, description=None, redo=False):

        if not isinstance(filter_values, list):
            filter_values = [filter_values]

        if result_name is None:
            result_name = name_to_filter + '_over_' + filter_name

        is_filter_and_to_filter_same_index_names = self.__is_same_index_names(filter_name, name_to_filter)

        if is_filter_and_to_filter_same_index_names:
            obj_filtered = Filters().filter_with_value(
                obj=self.get_data_object(name_to_filter), filter_obj=self.get_data_object(filter_name),
                filter_values=filter_values, result_name=result_name, xname=xname, yname=yname,
                label=label, xlabel=xlabel, ylabel=ylabel, category=category, description=description)
            self.add_object(result_name, obj_filtered, redo)
        else:
            raise TypeError(
                'Filter can not be applied on ' + name_to_filter + ' or ' + filter_name + ' is not a filter')

        return result_name

    def filter_with_time_occurrences(
            self, name_to_filter, filter_name,
            result_name=None, label=None, category=None, description=None, replace=False):

        if result_name is None:
            result_name = name_to_filter + '_over_' + filter_name

        name_to_filter_can_be_filtered = self.__is_indexed_by_exp_ant_frame(name_to_filter)
        name_filter_can_be_a_filter = self.__is_indexed_by_exp_ant_frame(filter_name)

        if name_to_filter_can_be_filtered and name_filter_can_be_a_filter:
            obj_filtered = Filters().filter(
                obj=self.get_data_object(name_to_filter), filter_obj=self.get_data_object(filter_name),
                name=result_name, label=label, category=category,
                description=description)
            self.add_object(result_name, obj_filtered, replace)
        else:
            raise TypeError(
                'Filter can not be applied on ' + name_to_filter + ' or ' + filter_name + ' is not a filter')

        return result_name

    def filter_with_time_intervals(
            self, name_to_filter, name_intervals, result_name=None,
            category=None, label=None, xlabel=None, ylabel=None, description=None,
            replace=True
    ):

        if result_name is None:
            result_name = name_to_filter + '_over_' + name_intervals

        name_to_filter_can_be_filtered = self.__is_indexed_by_exp_ant_frame(name_to_filter)
        name_interval_can_be_a_filter = self.get_object_type(name_intervals) == 'Events1d'

        interval_index_name = 'interval_idx_' + result_name
        if name_to_filter_can_be_filtered and name_interval_can_be_a_filter:

            self.__compute_time_interval_filter(
                name_to_filter, name_intervals, result_name, interval_index_name,
                category, label, xlabel, ylabel, description, replace)
        else:
            raise TypeError(
                name_to_filter + ' cannot be filtered (not indexed by exp,ant,frame or '
                + name_intervals + ' cannot be used as a filter (not an event)'
            )

        return result_name, interval_index_name

    def filter_with_experiment_characteristics(
            self, name_to_filter, chara_name, chara_values, result_name=None,
            xname=None, yname=None,
            category=None, label=None, xlabel=None, ylabel=None, description=None,
            replace=True
    ):

        if result_name is None:
            result_name = name_to_filter + '_over_' + chara_name

        if not isinstance(chara_values, list):
            chara_values = [chara_values]

        name_to_filter_can_be_filtered = self.__is_indexed_by_exp_ant_frame(name_to_filter)
        chara_name_can_be_a_filter = self.get_object_type(chara_name) == 'Characteristics1d'

        if name_to_filter_can_be_filtered and chara_name_can_be_a_filter:

            self.add_copy1d(
                name_to_copy=chara_name, copy_name='filter',
                category=category, label=label, description=description, replace=replace
            )
            self.operation('filter', lambda x: x == chara_values[0])
            for val in chara_values[1:]:
                self.operation('filter', lambda x: x == val)
            self.filter.df[self.filter.df == 0] = np.nan

            if self.__is_1d(name_to_filter):
                self.add_copy1d(
                    name_to_copy=name_to_filter, copy_name=result_name,
                    category=category, label=label, description=description, replace=replace
                )
            else:
                if xname is None:
                    xname = self.get_xname(name_to_filter)
                if yname is None:
                    yname = self.get_yname(name_to_filter)
                self.add_copy2d(
                    name_to_copy=name_to_filter, copy_name=result_name,
                    new_xname=xname, new_yname=yname,
                    category=category, label=label, xlabel=xlabel, ylabel=ylabel,
                    description=description, replace=replace
                )

            self.operation_between_2names(
                name1=result_name, name2='filter', func=lambda x, y: x * y
            )

            self.__dict__[result_name].df = self.__dict__[result_name].df.dropna()

            self.remove_object('filter')

    def __compute_time_interval_filter(
            self, name_to_filter, name_intervals, result_name, interval_index_name,
            category, label, xlabel, ylabel, description, replace):

        self.add_new_empty_basing_on_model(
            model_name=name_to_filter, result_name=result_name,
            category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description, replace=replace)

        self.add_new1d_empty(
            name=interval_index_name, object_type=self.get_object_type_in_1d(result_name),
            category=category, label=label, description=description, replace=replace)

        intervals_array = self.get_data_object(name_intervals).convert_df_to_array()
        for id_exp, id_ant, t, dt in intervals_array:
            temp_df = self.get_data_object(name_to_filter).get_row_of_id_exp_ant_in_frame_interval(
                id_exp, id_ant, t, t + dt)

            temp_interval_idx_df = pd.DataFrame(temp_df.copy()[temp_df.columns[0]])
            temp_interval_idx_df.columns = [self.get_columns(interval_index_name)]
            temp_interval_idx_df.iloc[:] = t

            self.get_data_object(result_name).add_df_as_rows(temp_df, replace=replace)

            self.get_data_object(interval_index_name).add_df_as_rows(temp_interval_idx_df, replace=replace)
        self.get_data_object(interval_index_name).df = self.get_df(interval_index_name).astype(int)

    def add_new_empty_basing_on_model(
            self, model_name, result_name,
            category=None, label=None, xlabel=None, ylabel=None, description=None, replace=False):

        name_to_filter_is_1d = self.__is_1d(model_name)
        if name_to_filter_is_1d:
            self.add_new1d_empty(
                name=result_name, object_type='Events1d',
                category=category, label=label, description=description, replace=replace
            )
        else:
            xname = self.get_xname(model_name)
            yname = self.get_yname(model_name)
            self.add_new2d_empty(
                name=result_name, xname=xname, yname=yname, object_type='Events2d',
                category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description, replace=replace
            )

    def operation(self, name, func):
        self.get_data_object(name).operation(func)

    def operation_between_2names(self, name1, name2, func, col_name1=None, col_name2=None):

        if self.__is_1d(name1) and self.__is_1d(name2):
            self.get_data_object(name1).operation_with_data_obj(obj=self.get_data_object(name2), func=func)
        elif self.__is_1d(name1) and not self.__is_1d(name2):
            self.get_data_object(name1).operation_with_data_obj(
                obj=self.get_data_object(name2), func=func, obj_name_col=col_name1)
        elif not self.__is_1d(name1) and self.__is_1d(name2):
            self.get_data_object(name1).operation_with_data_obj(
                obj=self.get_data_object(name2), func=func, self_name_col=col_name2)
        else:
            self.get_data_object(name1).operation_with_data_obj(
                obj=self.get_data_object(name2), func=func, self_name_col=col_name1, obj_name_col=col_name2)

    def event_extraction_from_timeseries(
            self, name_ts, name_extracted_events,
            label=None, category=None, description=None, replace=False):

        if self.get_object_type(name_ts) == 'TimeSeries1d':
            event = self.get_data_object(name_ts).extract_event(
                name=name_extracted_events, category=category, label=label, description=description)

            self.add_object(name_extracted_events, event, replace)

    def hist1d(self, name_to_hist, result_name=None, column_to_hist=None, bins='fd',
               category=None, label=None, description=None, replace=False):

        if result_name is None:
            result_name = name_to_hist + '_hist'

        if category is None:
            category = self.get_category(name_to_hist)

        df = self.get_data_object(name_to_hist).hist1d(column_name=column_to_hist, bins=bins)

        self.add_new1d_from_df(df=df, name=result_name, object_type=dataset_name,
                               category=category, label=label, description=description, replace=replace)

        return result_name

    def compute_time_delta(
            self, name_to_delta, result_name=None, xname=None, yname=None,
            category=None, label=None, xlabel=None, ylabel=None, description=None, replace=False
    ):

        if self.__is_indexed_by_exp_ant_frame(name_to_delta):

            if result_name is None:
                result_name = 'time_delta_' + name_to_delta
            if category is None:
                category = self.get_category(name_to_delta)

            delta_df = self.get_data_object(name_to_delta).compute_time_delta()

            if self.__is_1d(name_to_delta):
                self.add_new1d_from_df(
                    df=delta_df, name=result_name, object_type=self.get_object_type(name_to_delta),
                    category=category, label=label, description=description, replace=replace
                )
            else:
                if xname is None:
                    xname = self.get_xname(name_to_delta)
                if yname is None:
                    yname = self.get_yname(name_to_delta)
                self.add_new2d_from_df(
                    df=delta_df, name=result_name, xname=xname, yname=yname,
                    object_type=self.get_object_type(name_to_delta),
                    category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description,
                    replace=replace
                )

            return result_name
        else:
            raise TypeError(name_to_delta+' is not a time series')

    def compute_time_intervals(
            self, name_to_intervals, result_name=None, category=None, label=None, description=None, replace=False):

        if self.get_object_type(name_to_intervals) == 'TimeSeries1d':

            if result_name is None:
                result_name = name_to_intervals+'_intervals'

            if category is None:
                category = self.get_category(name_to_intervals)

            def interval4each_group(df: pd.DataFrame):
                df.iloc[:-1, :] = np.array(df.iloc[1:, :]) - np.array(df.iloc[:-1, :])
                df.iloc[-1, -1] = 0
                frame0 = df.index.get_level_values(id_frame_name)[0]

                arr = np.array(df[df != 0].dropna().reset_index())[:, 2:]
                df[:] = np.nan
                frame_list = arr[:, 0].copy().astype(int)
                if len(arr) != 0:
                    arr[1:, 0] = arr[1:, 0]-arr[:-1, 0]
                    arr[0, 0] -= frame0

                    inters = arr[arr[:, -1] == -1, :]
                    frame_list = frame_list[arr[:, -1] == -1]

                    df.loc[pd.IndexSlice[:, :, list(frame_list.astype(int))], :] = inters[:, 0]

                return df

            df_intervals = self.get_df(name_to_intervals).groupby([id_exp_name, id_ant_name]).apply(interval4each_group)
            df_intervals.dropna(inplace=True)
            df_intervals.astype(int, inplace=True)

            self.add_new1d_from_df(
                df=df_intervals, name=result_name, object_type='Events1d',
                category=category, label=label, description=description, replace=replace
            )

            self.load('fps')
            self.operation_between_2names(name1=result_name, name2='fps', func=lambda x, y: x/y)
            return result_name

        else:
            raise TypeError(name_to_intervals+' is not 1d time series')

    def compute_individual_mean(
            self, name_to_average, result_name=None, xname=None, yname=None,
            category=None, label=None, xlabel=None, ylabel=None, description=None, replace=False):

        if result_name is None:
            result_name = 'indiv_mean_' + name_to_average

        mean_df = self.get_data_object(name_to_average).mean_over_ants()

        if self.__is_1d(name_to_average):
            self.add_new1d_from_df(
                mean_df, result_name, 'AntCharacteristics1d',
                category=category, label=label, description=description, replace=replace)
        else:
            if xname is None:
                xname = self.get_xname(name_to_average)
            if yname is None:
                yname = self.get_yname(name_to_average)

            self.add_new2d_from_df(
                mean_df, result_name, object_type='AntCharacteristics2d',
                xname=xname, yname=yname, category=category,
                label=label, xlabel=xlabel, ylabel=ylabel, description=description, replace=replace)

    def compute_mean_in_time_intervals(
            self, name_to_average, name_intervals, mean_level=None, result_name=None,
            xname=None, yname=None, category=None,
            label=None, xlabel=None, ylabel=None, description=None, replace=False):

        if result_name is None:
            result_name = name_to_average + '_mean_over_' + name_intervals

        temp_name, interval_index_name = self.filter_with_time_intervals(
            name_to_filter=name_to_average,
            name_intervals=name_intervals,
            result_name='temp_obj')

        mean_df = self.get_data_object(temp_name).mean_over(
            self.get_df(interval_index_name), mean_level=mean_level, new_level_as=id_frame_name)

        self.remove_object([temp_name, interval_index_name])

        if self.__is_1d(name_to_average):
            self.add_new1d_from_df(
                mean_df, result_name, 'Events1d',
                category=category, label=label, description=description, replace=replace)
        else:
            if xname is None:
                xname = self.get_xname(name_to_average)
            if yname is None:
                yname = self.get_yname(name_to_average)

            self.add_new2d_from_df(
                mean_df, result_name, object_type='Events2d',
                xname=xname, yname=yname, category=category,
                label=label, xlabel=xlabel, ylabel=ylabel, description=description, replace=replace)

        return result_name

    def fit(self,
            name_to_fit, result_name=None, level=None, typ='linear', filter_name=None, filter_as_frame=False,
            window=None, sqrt_x=False, sqrt_y=False, normed=False, list_id_exp=None,
            xname=None, yname=None, category=None,
            label=None, xlabel=None, ylabel=None, description=None, replace=False):

        if result_name is None:
            result_name = name_to_fit + '_fit'
        if xname is None:
            xname = self.get_xname(name_to_fit)
        if yname is None:
            yname = self.get_yname(name_to_fit)

        df_filter = self.get_df(filter_name)

        df_fit = self.get_data_object(name_to_fit).fit(
            level=level, typ=typ, filter_df=df_filter,
            window=window, sqrt_x=sqrt_x, sqrt_y=sqrt_y, normed=normed, list_id_exp=list_id_exp
        )

        if filter_name is None:
            if level is None:
                return df_fit
            elif level == 'exp':
                self.add_new2d_from_df(
                    df=df_fit, name=result_name, xname=xname, yname=yname, object_type='Characteristics2d',
                    category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description,
                    replace=replace
                )
                return result_name
            elif level == 'ant':
                self.add_new2d_from_df(
                    df=df_fit, name=result_name, xname=xname, yname=yname, object_type='AntCharacteristics2d',
                    category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description,
                    replace=replace
                )
                return result_name
        else:
            if level is None:
                return df_fit
            else:
                if filter_as_frame is True:
                    self.pandas_index_manager.rename_index_level(df_fit, 'filter', id_frame_name)
                    if level == 'ant':
                        self.add_new2d_from_df(
                            df=df_fit, name=result_name, xname=xname, yname=yname, object_type='Events2d',
                            category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description,
                            replace=replace)
                        return result_name
                    else:
                        return df_fit

    def moving_mean(
            self, name_to_average, time_window, result_name=None,
            category=None, label=None, description=None, replace=False, segmented_computation=False):

        if not self.__is_1d(name_to_average):
            raise TypeError('moving mean not implemented for 2d')

        elif not self.__is_indexed_by_exp_ant_frame(name_to_average):
            raise TypeError('moving mean not implemented for if not indexed by (exp, ant, frame)')

        else:
            return self.moving_mean4exp_ant_frame_indexed_1d(
                name_to_average, time_window,
                result_name=result_name, category=category, label=label, description=description,
                replace=replace, segmented_computation=segmented_computation)

    def moving_mean4exp_ant_frame_indexed_1d(
            self, name_to_average, time_window, result_name=None,
            category=None, label=None, description=None, replace=False, segmented_computation=False):

        if not self.__is_1d(name_to_average):
            raise TypeError(name_to_average + ' is not 1d')

        elif not self.__is_indexed_by_exp_ant_frame(name_to_average):
            raise TypeError(name_to_average + ' is not indexed by (exp, ant, frame)')

        else:

            if result_name is None:
                result_name = 'mm' + str(time_window) + '_' + name_to_average

            if category is None:
                category = self.get_category(name_to_average)

            if label is None:
                label = 'MM of ' + name_to_average + ' on ' + str(time_window) + ' frames'

            if description is None:
                description = 'Moving mean of ' + name_to_average + ' on a time window of ' + str(
                    time_window) + ' frames'

            self.add_copy(
                old_name=name_to_average, new_name=result_name, category=category,
                label=label, description=description, replace=replace
            )
            self.get_df(result_name).dropna(inplace=True)

            time_window = int(np.floor(time_window / 2) * 2 + 1)

            def mm4each_group(df: pd.DataFrame):
                name = df.columns[0]

                if len(df) > 0:
                    id_exp = df.index.get_level_values(id_exp_name)[0]
                    id_ant = df.index.get_level_values(id_ant_name)[0]
                    frame0 = df.index.get_level_values(id_frame_name)[0]
                    frame1 = df.index.get_level_values(id_frame_name)[-1]

                    rg = range(frame0, frame1 + 1)
                    time_lg = frame1 - frame0 + 1

                    idx = pd.MultiIndex.from_tuples(
                        list(zip(np.full(time_lg, id_exp), np.full(time_lg, id_ant), rg)),
                        names=[id_exp_name, id_ant_name, id_frame_name])

                    df2 = df.reindex(idx)
                    time_array = np.array((1 - df2.isna()).astype(float))
                    mask0 = np.where(time_array == 0)[0]
                    mask1 = np.where(time_array == 1)[0]
                    values_array = np.array(df2[name])
                    values_array[mask0] = 0
                    mm_val = running_mean(values_array, time_window)[mask1]
                    mm_time = running_mean(time_array, time_window)[mask1]

                    df[name] = np.around(mm_val / mm_time, 3)

                return df

            if segmented_computation is True or len(self.get_df(result_name)) > 2e6:
                lg = len(set(self.get_df(result_name).index.get_level_values(id_exp_name)))
                lg = int(np.floor(lg / 5) * 5)
                for i in range(5, lg, 5):
                    idx_sl = pd.IndexSlice[i - 5:i, :, :]
                    self.__dict__[result_name].df.loc[idx_sl, :] = \
                        self.get_df(result_name).loc[idx_sl, :].groupby([id_exp_name, id_ant_name]).apply(mm4each_group)

                idx_sl = pd.IndexSlice[lg:, :, :]
                self.__dict__[result_name].df.loc[idx_sl, :] = \
                    self.get_df(result_name).loc[idx_sl, :].groupby([id_exp_name, id_ant_name]).apply(mm4each_group)

            else:
                self.__dict__[result_name].df \
                    = self.get_df(result_name).groupby([id_exp_name, id_ant_name]).apply(mm4each_group)

        return result_name

    def convert_xy_to_movie_system(self, id_exp, x, y):
        self.load(['food_center', 'mm2px', 'traj_translation', 'traj_reoriented'])

        if int(self.traj_reoriented.df.loc[id_exp]) == 1:
            x *= -1
            y *= -1

        x *= np.array(self.mm2px.df.loc[id_exp])
        y *= np.array(self.mm2px.df.loc[id_exp])

        x += np.array(self.food_center.df.x.loc[id_exp])
        y += np.array(self.food_center.df.y.loc[id_exp])

        x += np.array(self.traj_translation.df.x.loc[id_exp])
        y += np.array(self.traj_translation.df.y.loc[id_exp])

        return x, y

    def convert_xy_to_trajectory_system(self, id_exp, x, y):
        self.load(['food_center', 'mm2px', 'traj_translation', 'traj_reoriented'])

        x -= np.array(self.traj_translation.df.x.loc[id_exp])
        y -= np.array(self.traj_translation.df.y.loc[id_exp])

        x -= np.array(self.food_center.df.x.loc[id_exp])
        y -= np.array(self.food_center.df.y.loc[id_exp])

        x /= np.array(self.mm2px.df.loc[id_exp])
        y /= np.array(self.mm2px.df.loc[id_exp])

        if int(self.traj_reoriented.df.loc[id_exp]) == 1:
            x *= -1
            y *= -1

        return x, y

    def convert_xy_to_traj_system4each_group(self, df: pd.DataFrame):
        id_exp = df.index.get_level_values('id_exp')[0]
        x, y = self.convert_xy_to_trajectory_system(id_exp, df.x, df.y)
        df.x = x
        df.y = y
        return df
    
    def convert_orientation_to_movie_system(self, id_exp, orientation):
        self.load(['traj_reoriented'])
        if int(self.traj_reoriented.df.loc[id_exp]) == 1:
            orientation = norm_angle_tab(orientation - np.pi)
        return orientation

    def plot(self, name_to_plot, preplot=None, **kwargs):
        plotter = Plotter(root=self.root, obj=self.get_data_object(name_to_plot), **kwargs)
        return plotter.plot(preplot=preplot, **kwargs)
