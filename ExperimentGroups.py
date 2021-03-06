import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Tools.MiscellaneousTools.Geometry as Geo
from cv2 import cv2

from DataStructure.DataManager.DataFileManager import DataFileManager
from DataStructure.Builders.Builder import Builder
from DataStructure.DataObjects.CharacteristicEvents2d import CharacteristicEvents2dBuilder
from DataStructure.DataObjects.CharacteristicTimeSeries2d import CharacteristicTimeSeries2dBuilder
from DataStructure.DataObjects.Events2d import Events2dBuilder
from DataStructure.DataObjects.Filters import Filters
from DataStructure.DataObjects.TimeSeries2d import TimeSeries2dBuilder
from DataStructure.VariableNames import dataset_name, id_exp_name, id_ant_name, id_frame_name
from Movies.Movies import Movies
from Scripts.root import root_movie
from Tools.MiscellaneousTools.ArrayManipulation import turn_to_list
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager
from Tools.Plotter.Plotter import Plotter


class ExperimentGroups:

    def __init__(self, root, group):
        self.root = root + group + '/'
        self.group = group
        self.data_manager = DataFileManager(root, group)
        self.pandas_index_manager = PandasIndexManager()

        self.ref_id_exp = np.array(pd.read_csv(self.root + 'ref_id_exp.csv'))

        self.id_exp_list = list(set(self.ref_id_exp[:, 0]))
        self.id_exp_list.sort()

        self.names = set()

    @staticmethod
    def turn_to_list(names):
        if isinstance(names, str):
            names = [names]
        return names

    def get_data_object(self, name):
        return self.__dict__[name]

    def get_columns(self, name):
        return list(self.get_data_object(name).df.columns)

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

    def get_nb_indexes(self, name):
        return self.get_data_object(name).definition.nb_indexes

    def get_df(self, name):
        return self.get_data_object(name).df

    def get_array(self, name):
        return self.get_data_object(name).get_array()

    def get_level_values(self, name, id_name):
        return self.get_index(name).get_level_values(id_name)

    def get_index(self, name):
        return self.get_df(name).index

    def get_index_level_value(self, name, index_name):
        return self.get_df(name).index.get_level_values(index_name)

    def get_index_names(self, name):
        return list(self.get_index(name).names)

    def get_value(self, name, idx, name_col=None):
        if name_col is None:
            name_col = 0
        return self.get_data_object(name).df.loc[idx][name_col]

    def change_df(self, name, df):
        self.get_data_object(name).df = df

    def change_value(self, name, idx, value):
        self.get_data_object(name).df.loc[idx] = value

    def change_values(self, name, arr):
        self.get_data_object(name).change_values(arr)

    def groupby(self, name, index_to_group_with, func):
        return self.get_df(name).groupby(index_to_group_with).apply(func)

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

    def get_dict_id_exp_ant_frame(self, name):
        if self.get_object_type(name) in ['TimeSeries1d', 'TimeSeries2d']:
            return self.timeseries_exp_ant_frame_index
        else:
            return self.pandas_index_manager.get_index_dict(
                df=self.get_df(name), index_names=[id_exp_name, id_ant_name, id_frame_name])

    def is_name_existing(self, name):
        return name in self.data_manager.data_loader.definition_loader.definition_dict

    def plot_traj_on_movie(self, traj_names, id_exp, frame, id_ants=None):
        # TODO: rewrite this method with groupby
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

    def load(self, names, reload=True):
        names = self.turn_to_list(names)
        for name in names:
            if reload or name not in self.__dict__.keys():
                print('loading ', name)
                self.add_object(name, self.data_manager.load(name), replace=True)

    def write(self, names, modify_index=False):
        names = self.turn_to_list(names)
        for name in names:
            print('writing ', name)
            self.data_manager.write(self.get_data_object(name), modify_index=modify_index)

    def delete_data(self, names):
        names = self.turn_to_list(names)
        self.load(names)
        for name in names:
            print('deleting ', name)
            self.data_manager.delete(self.get_data_object(name))
            self.remove_object(name)

    def is_name_in_data(self, name):
        return self.data_manager.is_name_in_data(name)

    def load_as_2d(
            self, name1, name2, result_name, xname=None, yname=None, category=None,
            label=None, xlabel=None, ylabel=None, description=None, replace=False, reload=True):

        self.load([name1, name2], reload=reload)

        if self.__is_a_time_series(name1) and self.__is_a_time_series(name2):
            self.add_2d_from_1ds(
                name1=name1, name2=name2, result_name=result_name,
                xname=xname, yname=yname, category=category,
                label=label, xlabel=xlabel, ylabel=ylabel, description=description, replace=replace)

    def __is_1d(self, name):
        object_type = self.get_object_type(name)
        if object_type in ['Events2d', 'TimeSeries2d', 'Characteristics2d', 'CharacteristicTimeSeries2d',
                           'CharacteristicEvents2d']:
            return False
        elif object_type in [
                'Events1d', 'TimeSeries1d', 'Characteristics1d', 'AntCharacteristics1d',
                'CharacteristicTimeSeries1d', 'CharacteristicEvents1d']:
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

    def rename_category(self, old_name, new_name):
        self.data_manager.rename_category(old_name, new_name)

    def rename_in_exp(
            self, old_name, new_name=None, xname=None, yname=None, category=None,
            label=None, xlabel=None, ylabel=None, description=None):

        self.load(old_name)
        print('renaming ', old_name, 'to', new_name)

        if self.__is_1d(old_name) or self.get_object_type(old_name) == dataset_name:
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

        if self.__is_1d(old_name) or self.get_object_type(old_name) == dataset_name:
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
            elif object_type1 == 'CharacteristicEvents1d':
                event = CharacteristicEvents2dBuilder().build_from_1d(
                    event1=self.get_data_object(name1),
                    event2=self.get_data_object(name2), name=result_name, xname=xname,
                    yname=yname, category=category, label=label, xlabel=xlabel,
                    ylabel=ylabel, description=description)
                self.add_object(result_name, event, replace)
            else:
                raise TypeError(object_type1 + ' can not be gathered in 2d')
        else:
            raise TypeError(name1+' and '+name2+' do not have same index')

    def add_2d_from_2dfs(
            self, df1, df2, result_name, object_type, xname=None, yname=None,
            category=None, label=None, xlabel=None, ylabel=None, description=None, replace=False):

        self.add_new1d_from_df(df1, name='temp1', object_type=object_type)
        self.add_new1d_from_df(df2, name='temp2', object_type=object_type)

        self.add_2d_from_1ds(name1='temp1', name2='temp2', result_name=result_name,
                             xname=xname, yname=yname, category=category, label=label,
                             xlabel=xlabel, ylabel=ylabel, description=description, replace=replace)

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
        if object_type == dataset_name:
            raise ValueError('Use add_new_empty_dataset for DataSets')
        else:
            df = self.__create_empty_df(name, object_type)
            obj = Builder.build1d_from_df(
                df=df, name=name, object_type=object_type, category=category, label=label, description=description)
            self.add_object(name, obj, replace=replace)

    def __create_empty_df(self, name, object_type, xname=None, yname=None, index_names=None):
        # TODO: for all, do like for object type characteristics1d
        if object_type in ['TimeSeries1d', 'TimeSeries2d', 'Events1d', 'Events2d']:
            df = self.pandas_index_manager.create_empty_df(column_names=name,
                                                           index_names=[id_exp_name, id_ant_name, id_frame_name])
        elif object_type in ['AntCharacteristics1d']:
            df = self.pandas_index_manager.create_empty_df(column_names=name, index_names=[id_exp_name, id_ant_name])
        elif object_type in ['Characteristics2d']:
            empty_column = np.full(len(self.id_exp_list), 0)
            empty_array = np.array(list(zip(self.id_exp_list, empty_column, empty_column)))
            df = pd.DataFrame(empty_array, columns=[id_exp_name, xname, yname])
            df.set_index([id_exp_name], inplace=True)
            df[:] = np.nan
        elif object_type in ['Characteristics1d']:
            df = pd.DataFrame(np.array(list(zip(self.id_exp_list, np.full(len(self.id_exp_list), 0)))),
                              columns=[id_exp_name, name])
            df.set_index([id_exp_name], inplace=True)
            df[:] = np.nan
        elif object_type in ['CharacteristicTimeSeries1d', 'CharacteristicTimeSeries2d',
                             'CharacteristicEvents1d', 'CharacteristicEvents2d']:
            df = self.pandas_index_manager.create_empty_df(column_names=name, index_names=[id_exp_name, id_frame_name])
        elif object_type in [dataset_name]:
            df = self.pandas_index_manager.create_empty_df(column_names=name, index_names=index_names)
        else:
            raise IndexError('Object type ' + object_type + ' unknown')
        return df

    def add_new2d_empty(
            self, name, xname, yname, object_type, category=None,
            label=None, xlabel=None, ylabel=None, description=None, replace=False):

        if object_type == dataset_name:
            raise ValueError('Use add_new_empty_dataset for DataSets')
        else:
            df = self.__create_empty_df(name=name, xname=xname, yname=yname, object_type=object_type)

            obj = Builder.build2d_from_df(
                df=df, name=name, xname=xname, yname=yname,
                object_type=object_type, category=category,
                label=label, xlabel=xlabel, ylabel=ylabel, description=description)

            self.add_object(name, obj, replace)

    def add_new1d_from_df(self, df, name, object_type, category=None, label=None, description=None, replace=False):
        if object_type == dataset_name:
            raise ValueError('Use add_new_dataset_from_df for DataSets')
        else:
            obj = Builder.build1d_from_df(
                df=pd.DataFrame(df), name=name, object_type=object_type,
                category=category, label=label, description=description)
            self.add_object(name, obj, replace)

    def add_new2d_from_df(
            self, df, name, xname, yname, object_type, category=None,
            label=None, xlabel=None, ylabel=None, description=None, replace=False):

        if object_type == dataset_name:
            raise ValueError('Use add_new_dataset_from_df for DataSets')
        else:
            obj = Builder.build2d_from_df(
                df=df, name=name, xname=xname, yname=yname,
                object_type=object_type, category=category,
                label=label, xlabel=xlabel, ylabel=ylabel, description=description)

            self.add_object(name, obj, replace=replace)

    def add_new_empty_dataset(
            self, name, index_names, column_names, index_values=None, fill_value=np.nan,
            category=None, label=None, description=None, replace=False):

        if index_values is None:
            df = PandasIndexManager.create_empty_df(index_names=index_names, column_names=column_names)
        else:
            index_names = turn_to_list(index_names)
            column_names = turn_to_list(column_names)

            empty_array = np.zeros((len(index_values), len(column_names)))
            df = pd.DataFrame(empty_array, columns=column_names)

            if len(index_names) == 1 or isinstance(index_values, pd.core.indexes.multi.MultiIndex):
                indexes = index_values
            else:
                indexes = pd.MultiIndex.from_arrays(np.array(index_values).T)

            df.index = indexes
            df.index.names = index_names

            df[:] = fill_value
            df = df.astype(type(fill_value))

        obj = Builder.build_dataset_from_df(
            df=df, name=name, category=category, label=label, description=description)
        self.add_object(name, obj, replace)

    def add_new_dataset_from_df(self, df, name, category=None, label=None, description=None, replace=False):

        obj = Builder.build_dataset_from_df(
            df=pd.DataFrame(df), name=name, category=category, label=label, description=description)
        self.add_object(name, obj, replace)

    def add_new1d_from_array(
            self, array, name, object_type, category=None, label=None, description=None, replace=False):

        df = self.__convert_array_to_df(array, name, object_type)

        obj = Builder.build1d_from_df(
            df=df, name=name, object_type=object_type, category=category, label=label, description=description)

        self.add_object(name, obj, replace=replace)

    def add_new_dataset_from_array(self, array, name, index_names, column_names=None, category=None, label=None,
                                   description=None, replace=False):

        if column_names is None:
            column_names = name

        df = self.__convert_array_to_df(array=array, name=column_names,
                                        object_type=dataset_name, index_names=index_names)

        obj = Builder.build_dataset_from_df(
            df=df, name=name, category=category, label=label, description=description)

        self.add_object(name, obj, replace=replace)

    def __convert_array_to_df(self, array, name, object_type, index_names=None):

        if object_type in ['TimeSeries1d', 'TimeSeries2d', 'Events1d', 'Events2d']:
            df = self.pandas_index_manager.convert_array_to_df(
                array, index_names=[id_exp_name, id_ant_name, id_frame_name], column_names=name)
            # df.index = pd.MultiIndex.from_arrays(
            #     np.array(list(df.index), dtype=int), names=[id_exp_name, id_ant_name, id_frame_name])

        elif object_type in ['AntCharacteristics1d', 'AntCharacteristics2d']:
            df = self.pandas_index_manager.convert_array_to_df(
                array, index_names=[id_exp_name, id_ant_name], column_names=name)
            # df.index = pd.MultiIndex.from_arrays(
            #     np.array(list(df.index), dtype=int), names=[id_exp_name, id_ant_name])

        elif object_type in ['Characteristics1d', 'Characteristics2d']:
            df = self.pandas_index_manager.convert_array_to_df(array, index_names=id_exp_name, column_names=name)
            # df.index = pd.MultiIndex.from_arrays(
            #     np.array(list(df.index), dtype=int), names=id_exp_name)

        elif object_type in ['CharacteristicTimeSeries1d', 'CharacteristicTimeSeries2d',
                             'CharacteristicEvents1d', 'CharacteristicEvents2d']:
            df = self.pandas_index_manager.convert_array_to_df(
                array, index_names=[id_exp_name, id_frame_name], column_names=name)
            # df.index = pd.MultiIndex.from_arrays(
            #     np.array(list(df.index), dtype=int), names=[id_exp_name, id_frame_name])

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
            xname=None, yname=None, category=None,
            label=None, xlabel=None, ylabel=None, description=None, replace=False):

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
            self.add_object(result_name, obj_filtered, replace)
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
                obj=self.get_data_object(name2), func=func, obj_name_col=col_name2)
        elif not self.__is_1d(name1) and self.__is_1d(name2):
            self.get_data_object(name1).operation_with_data_obj(
                obj=self.get_data_object(name2), func=func, self_name_col=col_name1)
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

    def sum_over_exp_and_frames(
            self, name_to_average, result_name=None, xname=None, yname=None,
            category=None, label=None, xlabel=None, ylabel=None, description=None, replace=False):

        if result_name is None:
            result_name = name_to_average+'_sum_exp_and_frames'

        sum_df = self.get_data_object(name_to_average).sum_over_exp_and_frames()

        if self.__is_1d(name_to_average):
            if self.get_object_type(name_to_average) == 'TimeSeries1d':
                self.add_new1d_from_df(
                    sum_df, result_name, 'CharacteristicTimeSeries1d',
                    category=category, label=label, description=description, replace=replace)
            else:
                self.add_new1d_from_df(
                    sum_df, result_name, 'CharacteristicEvents1d',
                    category=category, label=label, description=description, replace=replace)
        else:
            if xname is None:
                xname = self.get_xname(name_to_average)
            if yname is None:
                yname = self.get_yname(name_to_average)

            if self.get_object_type(name_to_average) is 'TimeSeries2d':
                self.add_new2d_from_df(
                    sum_df, result_name, object_type='CharacteristicTimeSeries2d',
                    xname=xname, yname=yname, category=category,
                    label=label, xlabel=xlabel, ylabel=ylabel, description=description, replace=replace)
            else:
                self.add_new2d_from_df(
                    sum_df, result_name, object_type='CharacteristicEvents2d',
                    xname=xname, yname=yname, category=category,
                    label=label, xlabel=xlabel, ylabel=ylabel, description=description, replace=replace)

    def mean_over_exp_and_frames(
            self, name_to_average, result_name=None, xname=None, yname=None,
            category=None, label=None, xlabel=None, ylabel=None, description=None, replace=False):

        if result_name is None:
            result_name = name_to_average+'_mean_over_exp_and_frames'

        mean_df = self.get_data_object(name_to_average).mean_over_exp_and_frames()

        if self.__is_1d(name_to_average):
            if self.get_object_type(name_to_average) is 'TimeSeries1d':
                self.add_new1d_from_df(
                    mean_df, result_name, 'CharacteristicTimeSeries1d',
                    category=category, label=label, description=description, replace=replace)
            else:
                self.add_new1d_from_df(
                    mean_df, result_name, 'CharacteristicEvents1d',
                    category=category, label=label, description=description, replace=replace)
        else:
            if xname is None:
                xname = self.get_xname(name_to_average)
            if yname is None:
                yname = self.get_yname(name_to_average)

            if self.get_object_type(name_to_average) is 'TimeSeries2d':
                self.add_new2d_from_df(
                    mean_df, result_name, object_type='CharacteristicTimeSeries2d',
                    xname=xname, yname=yname, category=category,
                    label=label, xlabel=xlabel, ylabel=ylabel, description=description, replace=replace)
            else:
                self.add_new2d_from_df(
                    mean_df, result_name, object_type='CharacteristicEvents2d',
                    xname=xname, yname=yname, category=category,
                    label=label, xlabel=xlabel, ylabel=ylabel, description=description, replace=replace)

    def hist1d(self, name_to_hist, result_name=None, column_to_hist=None, bins='fd', error=False,
               category=None, label=None, description=None, replace=False):

        if result_name is None:
            result_name = name_to_hist + '_hist'

        if category is None:
            category = self.get_category(name_to_hist)

        if label is None and not self.get_label(name_to_hist) is None:
            label = self.get_label(name_to_hist)+' histogram'

        if description is None and not self.get_description(name_to_hist) is None:
            description = 'Histogram of '+self.get_description(name_to_hist).lower()

        df = self.get_data_object(name_to_hist).hist1d(column_name=column_to_hist, bins=bins, error=error)

        self.add_new_dataset_from_df(df=df, name=result_name, category=category,
                                     label=label, description=description, replace=replace)

        return result_name

    def hist2d(self, xname_to_hist, yname_to_hist, result_name=None, xcolumn_to_hist=None, ycolumn_to_hist=None,
               bins=10, category=None, label=None, description=None, replace=False):

        if result_name is None:
            result_name = xname_to_hist+'_'+xname_to_hist + '_hist2d'

        if category is None:
            category = self.get_category(yname_to_hist)

        if label is None:
            if self.get_label(xname_to_hist) is not None and self.get_label(yname_to_hist) is not None:
                label = '2d histogram of '+self.get_label(xname_to_hist)+' and '+self.get_label(yname_to_hist)

        if description is None:
            if self.get_label(xname_to_hist) is not None and self.get_label(yname_to_hist) is not None:
                description = '2d histogram of '+self.get_label(xname_to_hist)+' and '+self.get_label(yname_to_hist)

        df = self.get_data_object(yname_to_hist).hist2d(self.get_df(xname_to_hist), bins=bins,
                                                        column_name=ycolumn_to_hist, column_name2=xcolumn_to_hist)

        self.add_new_dataset_from_df(df=df, name=result_name, category=category,
                                     label=label, description=description, replace=replace)

        return result_name

    def survival_curve(self, name, start=0, result_name=None, column_to_hist=None,
                       category=None, label=None, description=None, replace=False):

        if result_name is None:
            result_name = name + '_surv'

        if category is None:
            category = self.get_category(name)

        if label is None and not self.get_label(name) is None:
            label = self.get_label(name) + ' histogram'

        if description is None and not self.get_description(name) is None:
            description = 'Survival curve of '+self.get_description(name).lower()

        df = self.get_data_object(name).survival_curve(start=start, column_name=column_to_hist)

        self.add_new_dataset_from_df(df=df, name=result_name, category=category,
                                     label=label, description=description, replace=replace)

        return result_name

    def vs(self, xname, yname, result_name=None, xcolumn_to_hist=None, ycolumn_to_hist=None,
           n_bins=10, x_are_integers=False, category=None, label=None, description=None, replace=False):

        if result_name is None:
            result_name = yname+'_vs_'+xname

        if category is None:
            category = self.get_category(yname)

        if label is None:
            if self.get_label(xname) is not None and self.get_label(yname) is not None:
                label = self.get_label(xname)+' vs '+self.get_label(yname)

        if description is None:
            if self.get_label(xname) is not None and self.get_label(yname) is not None:
                description = self.get_label(xname)+' vs '+self.get_label(yname)

        df = self.get_data_object(yname).vs(self.get_df(xname), n_bins=n_bins, x_are_integers=x_are_integers,
                                            column_name=ycolumn_to_hist, column_name2=xcolumn_to_hist)

        self.add_new_dataset_from_df(df=df, name=result_name, category=category,
                                     label=label, description=description, replace=replace)

        return result_name

    def hist1d_evolution(
            self, name_to_hist, start_frame_intervals, end_frame_intervals, bins, index_name=None, normed=False,
            result_name=None, column_to_hist=None, category=None, label=None, description=None, replace=False,
            fps=100.):

        if result_name is None:
            result_name = name_to_hist + '_evol_hist'

        if category is None:
            category = self.get_category(name_to_hist)

        if label is None:
            if self.get_label(name_to_hist) is not None:
                label = self.get_label(name_to_hist) + ' histogram evolution'

        if description is None:
            if self.get_description(name_to_hist) is not None:
                description = 'Evolution of histogram of ' + self.get_description(name_to_hist).lower()

        if self.get_object_type(name_to_hist) == dataset_name:

            df = self.get_data_object(name_to_hist).hist1d_evolution(
                column_name=column_to_hist, index_name=index_name, start_frame_intervals=start_frame_intervals,
                end_frame_intervals=end_frame_intervals, bins=bins, normed=normed, fps=fps)

        elif self.__is_indexed_by_exp_ant_frame(name_to_hist) or self.__is_indexed_by_exp_frame(name_to_hist):

            df = self.get_data_object(name_to_hist).hist1d_evolution(
                column_name=column_to_hist, start_frame_intervals=start_frame_intervals,
                end_frame_intervals=end_frame_intervals, bins=bins, normed=normed)

        else:
            raise TypeError(name_to_hist+' is not frame indexed or not Dataset')

        self.add_new_dataset_from_df(df=df, name=result_name, category=category,
                                     label=label, description=description, replace=replace)

        return result_name

    def sum_evolution(
            self, name_to_var, start_index_intervals, end_index_intervals, index_name=None,
            result_name=None, column_to_var=None, category=None, label=None, description=None, replace=False):

        if result_name is None:
            result_name = name_to_var + '_sum_evol'

        if category is None:
            category = self.get_category(name_to_var)

        if label is None:
            if self.get_label(name_to_var) is not None:
                label = self.get_label(name_to_var) + ' sum evolution'

        if description is None:
            if self.get_description(name_to_var) is not None:
                description = 'Evolution of sum of ' + self.get_description(name_to_var).lower()

        if self.__is_indexed_by_exp_ant_frame(name_to_var) or self.__is_indexed_by_exp_frame(name_to_var):

            df = self.get_data_object(name_to_var).sum_evolution(
                column_name=column_to_var, start_frame_intervals=start_index_intervals,
                end_frame_intervals=end_index_intervals)

        elif self.get_object_type(name_to_var) == dataset_name:

            df = self.get_data_object(name_to_var).sum_evolution(
                column_name=column_to_var, index_name=index_name, start_index_intervals=start_index_intervals,
                end_index_intervals=end_index_intervals)

        else:
            raise TypeError(name_to_var+' is not dataset')

        df.columns = ['mean']

        self.add_new_dataset_from_df(df=df, name=result_name, category=category,
                                     label=label, description=description, replace=replace)

        return result_name

    def mean_evolution(
            self, name_to_var, start_index_intervals, end_index_intervals, error=None, index_name=None,
            result_name=None, column_to_var=None, category=None, label=None, description=None, replace=False):

        if result_name is None:
            result_name = name_to_var + '_mean_evol'

        if category is None:
            category = self.get_category(name_to_var)

        if label is None:
            if self.get_label(name_to_var) is not None:
                label = self.get_label(name_to_var) + ' mean evolution'

        if description is None:
            if self.get_description(name_to_var) is not None:
                description = 'Evolution of mean of ' + self.get_description(name_to_var).lower()

        if self.__is_indexed_by_exp_ant_frame(name_to_var) or self.__is_indexed_by_exp_frame(name_to_var):

            df = self.get_data_object(name_to_var).mean_evolution(
                column_name=column_to_var, error=error, start_frame_intervals=start_index_intervals,
                end_frame_intervals=end_index_intervals)

        elif self.get_object_type(name_to_var) == dataset_name:

            df = self.get_data_object(name_to_var).mean_evolution(
                column_name=column_to_var, index_name=index_name, error=error,
                start_index_intervals=start_index_intervals, end_index_intervals=end_index_intervals)

        else:
            raise TypeError(name_to_var+' is not dataset')

        if error is None:
            df.columns = ['mean']
        elif error is True:
            df.columns = ['mean', 'IC95_1', 'IC95_2']
        elif error == 'binomial':
            df.columns = ['mean', 'err1', 'err2']

        self.add_new_dataset_from_df(df=df, name=result_name, category=category,
                                     label=label, description=description, replace=replace)

        return result_name

    def variance_evolution(
            self, name_to_var, start_index_intervals, end_index_intervals, index_name=None,
            result_name=None, column_to_var=None, category=None, label=None, description=None, replace=False):

        if result_name is None:
            result_name = name_to_var + '_var_evol'

        if category is None:
            category = self.get_category(name_to_var)

        if label is None:
            if self.get_label(name_to_var) is not None:
                label = self.get_label(name_to_var) + ' variance evolution'

        if description is None:
            if self.get_description(name_to_var) is not None:
                description = 'Evolution of variance of ' + self.get_description(name_to_var).lower()

        if self.__is_indexed_by_exp_ant_frame(name_to_var) or self.__is_indexed_by_exp_frame(name_to_var):

            df = self.get_data_object(name_to_var).variance_evolution(
                column_name=column_to_var, start_frame_intervals=start_index_intervals,
                end_frame_intervals=end_index_intervals)

        elif self.get_object_type(name_to_var) == dataset_name:

            df = self.get_data_object(name_to_var).variance_evolution(
                column_name=column_to_var, index_name=index_name, start_index_intervals=start_index_intervals,
                end_index_intervals=end_index_intervals)

        else:
            raise TypeError(name_to_var+' is not dataset')

        df.columns = ['variance']

        self.add_new_dataset_from_df(df=df, name=result_name, category=category,
                                     label=label, description=description, replace=replace)

        return result_name

    def compute_time_intervals(
            self, name_to_intervals, result_name=None, category=None, label=None, description=None, replace=False):

        if result_name is None:
            result_name = name_to_intervals + '_intervals'

        if category is None:
            category = self.get_category(name_to_intervals)

        if self.get_object_type(name_to_intervals) == 'TimeSeries1d':

            object_type = 'Events1d'

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
                    if arr[0, 1] == -1:
                        arr = arr[1:, :]
                        frame_list = frame_list[1:]

                    inters = arr[arr[:, -1] == -1, 0]
                    frame_list = frame_list[arr[:, -1] == -1]-inters

                    frame_list = frame_list[1:]
                    inters = inters[1:]

                    df.loc[pd.IndexSlice[:, :, list(frame_list.astype(int))], :] = inters

                return df
            df_intervals = self.get_df(name_to_intervals).groupby([id_exp_name, id_ant_name]).apply(interval4each_group)

        elif self.get_object_type(name_to_intervals) == 'CharacteristicTimeSeries1d':
            object_type = 'CharacteristicEvents1d'

            def interval4each_group(df: pd.DataFrame):
                df.iloc[:-1, :] = np.array(df.iloc[1:, :]) - np.array(df.iloc[:-1, :])
                df.iloc[-1, -1] = 0
                frame0 = df.index.get_level_values(id_frame_name)[0]

                arr = np.array(df[df != 0].dropna().reset_index())[:, 1:]
                df[:] = np.nan
                frame_list = arr[:, 0].copy().astype(int)
                if len(arr) != 0:
                    arr[1:, 0] = arr[1:, 0]-arr[:-1, 0]
                    arr[0, 0] -= frame0

                    inters = arr[arr[:, -1] == -1, 0]
                    frame_list = frame_list[arr[:, -1] == -1]-inters

                    frame_list = frame_list[1:]
                    inters = inters[1:]

                    list_idx = list(frame_list.astype(int))
                    df.loc[pd.IndexSlice[:, list_idx], :] = np.c_[inters]

                return df
            df_intervals = self.get_df(name_to_intervals).groupby(id_exp_name).apply(interval4each_group)

        else:
            raise TypeError(name_to_intervals+' is not 1d time series')

        df_intervals.dropna(inplace=True)

        self.add_new1d_from_df(
            df=df_intervals, name=result_name, object_type=object_type,
            category=category, label=label, description=description, replace=replace
        )

        self.load('fps')
        self.operation_between_2names(name1=result_name, name2='fps', func=lambda x, y: x/y)
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

    def compute_individual_mean(
            self, name_to_average, result_name=None, xname=None, yname=None,
            category=None, label=None, xlabel=None, ylabel=None, description=None, replace=False):

        if result_name is None:
            result_name = 'indiv_mean_' + name_to_average

        mean_df = self.get_data_object(name_to_average).mean_over_exp_and_ants()

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

    def fit(self, name_to_fit, typ='linear', window=None,
            sqrt_x=False, sqrt_y=False, normed=False, column=None, cst=None):

        if self.get_object_type(name_to_fit) == dataset_name:

            fit = self.get_data_object(name_to_fit).fit(typ=typ, window=window, sqrt_x=sqrt_x, sqrt_y=sqrt_y,
                                                        normed=normed, column=column, cst=cst)
            return fit

        else:
            raise TypeError(name_to_fit + ' is not a dataset')

    def rolling_mean(
            self, name_to_average, window, is_angle, result_name=None, index_names=None,
            category=None, label=None, description=None, replace=False):

        if not self.__is_1d(name_to_average):
            raise TypeError('moving mean not implemented for 2d')

        else:
            if result_name is None:
                result_name = 'mm' + str(window) + '_' + name_to_average

            if category is None:
                category = self.get_category(name_to_average)

            if label is None:
                label = 'MM of ' + name_to_average + ' on ' + str(window) + ' frames'

            if description is None:
                description = 'Moving mean of ' + name_to_average + ' on a time window of ' + str(
                    window) + ' frames'

            if self.__is_indexed_by_exp_ant_frame(name_to_average) or self.__is_indexed_by_exp_frame(name_to_average):

                if is_angle is True:
                    df = self.get_data_object(name_to_average).rolling_mean_angle(window)
                else:
                    df = self.get_data_object(name_to_average).rolling_mean(window)
                df.columns = [result_name]
                self.add_new1d_from_df(df=df, name=result_name, object_type=self.get_object_type(name_to_average),
                                       category=category, label=label, description=description, replace=replace)

            elif self.get_object_type(name_to_average) == dataset_name:

                if is_angle is True:
                    df = self.get_data_object(name_to_average).rolling_mean_angle(window, index_names=index_names)
                else:
                    df = self.get_data_object(name_to_average).rolling_mean(window, index_names=index_names)
                df.columns = [result_name]
                self.add_new_dataset_from_df(df=df, name=result_name, category=category,
                                             label=label, description=description, replace=replace)

            else:
                raise TypeError('moving mean not implemented for '+self.get_object_type(name_to_average))
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
            orientation = Geo.norm_angle(orientation - np.pi)
        return orientation

    def plot(self, name_to_plot, preplot=None, **kwargs):
        plotter = Plotter(root=self.root, obj=self.get_data_object(name_to_plot), **kwargs)
        return plotter.plot(preplot=preplot, **kwargs)
