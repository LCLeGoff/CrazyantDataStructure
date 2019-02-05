import pandas as pd

import numpy as np

from DataStructure.DataManager.DataFileManager import DataFileManager
from DataStructure.DataObjectBuilders.Builder import Builder
from DataStructure.DataObjects.CharacteristicTimeSeries2d import CharacteristicTimeSeries2dBuilder
from DataStructure.DataObjects.Events2d import Events2dBuilder
from DataStructure.DataObjects.Filters import Filters
from DataStructure.DataObjects.TimeSeries2d import TimeSeries2dBuilder
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class ExperimentGroups:

    def __init__(self, root, group, id_exp_list):
        self.root = root
        self.group = group
        self.data_manager = DataFileManager(root, group)
        self.pandas_index_manager = PandasIndexManager()
        self.id_exp_list = id_exp_list
        self.id_exp_list.sort()
        self.names = set()

    @staticmethod
    def turn_to_list(names):
        if isinstance(names, str):
            names = [names]
        return names

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
        object_type = object_type[:-2]+'1d'
        return object_type

    def get_label(self, name):
        return self.get_data_object(name).definition.label

    def get_xlabel(self, name):
        return self.get_data_object(name).definition.xlabel

    def get_ylabel(self, name):
        return self.get_data_object(name).definition.ylabel

    def get_description(self, name):
        return self.get_data_object(name).definition.description

    def get_data_object(self, name):
        return self.__dict__[name]

    def get_df(self, name):
        return self.get_data_object(name).df

    def get_index_names(self, name):
        return self.get_df(name).index.names

    def set_id_exp_list(self, id_exp_list=None):
        if id_exp_list is None:
            id_exp_list = self.id_exp_list
        return id_exp_list

    def add_object(self, name, obj, replace=False):
        if not replace and name in self.names:
            raise NameError(name+' exists already')
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
            print('loading ', name)
            if name not in self.__dict__.keys():
                self.add_object(name, self.data_manager.load(name), replace=True)

    def write(self, names):
        names = self.turn_to_list(names)
        for name in names:
            print('writing ', name)
            self.data_manager.write(self.get_data_object(name))

    def delete_data(self, names):
        names = self.turn_to_list(names)
        for name in names:
            print('deleting ', name)
            self.data_manager.delete(self.get_data_object(name))

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
        else:
            raise TypeError('Object type ' + object_type + ' unknown')

    def __is_indexed_by_exp_ant_frame(self, name):
        object_type = self.get_object_type(name)
        return object_type in ['Events1d', 'TimeSeries1d', 'Events2d', 'TimeSeries2d']

    def __is_same_index_names(self, name1, name2):
        index_name1 = self.get_index_names(name1)
        index_name2 = self.get_index_names(name2)
        return set(index_name1) == set(index_name2)

    def rename_data(self, old_name, new_name):
        self.load(old_name)
        self.rename_object(old_name, new_name)
        self.delete_data(old_name)
        self.remove_object(old_name)
        self.write(new_name)

    def rename_object(self, old_name, new_name, replace=False):
        if self.__is_1d(old_name):
            self.add_copy1d(name_to_copy=old_name, copy_name=new_name, copy_definition=True, replace=replace)
        else:
            self.add_copy2d(name_to_copy=old_name, copy_name=new_name, copy_definition=True, replace=replace)

    def add_2d_from_1ds(
            self, name1, name2, result_name, xname, yname,
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
        df = self._create_empty_df(name, object_type)
        obj = Builder.build1d_from_df(
            df=df, name=name, object_type=object_type, category=category, label=label, description=description)
        self.add_object(name, obj, replace=replace)

    def _create_empty_df(self, name, object_type):
        if object_type in ['Events1d', 'TimeSeries1d']:
            df = self.pandas_index_manager.create_empty_exp_ant_frame_indexed_1d_df(name)
        elif object_type in ['AntCharacteristics1d']:
            df = self.pandas_index_manager.create_empty_exp_ant_indexed_df(name)
        elif object_type in ['Characteristics1d']:
            df = self.pandas_index_manager.create_empty_exp_indexed_1d_df(name)
        elif object_type in ['Events2d', 'TimeSeries2d']:
            df = self.pandas_index_manager.create_empty_exp_ant_frame_indexed_2d_df(name[0], name[1])
        elif object_type in ['Characteristics2d']:
            df = self.pandas_index_manager.create_empty_exp_indexed_2d_df(name[0], name[1])
        elif object_type in ['CharacteristicTimeSeries1d']:
            df = self.pandas_index_manager.create_empty_exp_frame_indexed_1d_df(name)
        elif object_type in ['CharacteristicTimeSeries2d']:
            df = self.pandas_index_manager.create_empty_exp_frame_indexed_2d_df(name[0], name[1])
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

        df = self.__convert_array_to_1d_df(array, name, object_type)

        obj = Builder.build1d_from_df(
            df=df, name=name, object_type=object_type, category=category, label=label, description=description)

        self.add_object(name, obj, replace=replace)

    def __convert_array_to_1d_df(self, array, name, object_type):
        if not self.__is_1d(name):
            raise TypeError('Object in 2d')
        elif object_type in ['Events1d', 'TimeSeries1d']:
            df = self.pandas_index_manager.convert_to_exp_ant_frame_indexed_1d_df(array, name)
        elif object_type in ['AntCharacteristics1d']:
            df = self.pandas_index_manager.convert_to_exp_ant_indexed_df(array, name)
        elif object_type in ['Characteristics1d']:
            df = self.pandas_index_manager.convert_to_exp_indexed_df(array, name)
        elif object_type in ['CharacteristicTimeSeries1d']:
            df = self.pandas_index_manager.convert_to_exp_frame_indexed_1d_df(array, name)
        else:
            raise IndexError('Object type ' + object_type + ' unknown')
        return df

    def filter_with_values(
            self, name_to_filter, filter_name, filter_values=1, result_name=None,
            xname=None, yname=None, label=None, xlabel=None, ylabel=None,
            category=None, description=None, redo=False):

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

        interval_index_name = 'interval_idx_'+result_name
        if name_to_filter_can_be_filtered and name_interval_can_be_a_filter:

            self._compute_time_interval_filter(
                name_to_filter, name_intervals, result_name, interval_index_name,
                category, label, xlabel, ylabel, description, replace)

        # self.rename_object('temp', result_name, replace)
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
                name1=result_name, name2='filter', fct=lambda x, y: x*y
            )

            self.__dict__[result_name].df = self.__dict__[result_name].df.dropna()

            self.remove_object('filter')

    def _compute_time_interval_filter(
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

    def operation(self, name, fct):
        self.get_data_object(name).operation(fct)

    def operation_between_2names(self, name1, name2, fct, col_name1=None, col_name2=None):

        if self.__is_1d(name1) and self.__is_1d(name2):
            self.get_data_object(name1).operation_with_data_obj(obj=self.get_data_object(name2), fct=fct)
        elif self.__is_1d(name1) and not self.__is_1d(name2):
            self.get_data_object(name1).operation_with_data_obj(
                obj=self.get_data_object(name2), fct=fct, obj_name_col=col_name1)
        elif not self.__is_1d(name1) and self.__is_1d(name2):
            self.get_data_object(name1).operation_with_data_obj(
                obj=self.get_data_object(name2), fct=fct, self_name_col=col_name2)
        else:
            self.get_data_object(name1).operation_with_data_obj(
                obj=self.get_data_object(name2), fct=fct, self_name_col=col_name1, obj_name_col=col_name2)

    def event_extraction_from_timeseries(
            self, name_ts, name_extracted_events,
            label=None, category=None, description=None, replace=False):

        if self.get_object_type(name_ts) == 'TimeSeries1d':
            event = self.get_data_object(name_ts).extract_event(
                name=name_extracted_events, category=category, label=label, description=description)

            self.add_object(name_extracted_events, event, replace)

    def compute_delta(
            self, name_to_delta, filter_name=None, result_name=None, xname=None, yname=None,
            category=None, label=None, xlabel=None, ylabel=None, description=None, replace=False
    ):
        if filter_name is None:
            filter_obj = None
        else:
            filter_obj = self.get_data_object(filter_name)
        if result_name is None:
            result_name = 'delta_'+self.get_df(name_to_delta).columns[0]
        if category is None:
            category = self.get_category(name_to_delta)

        delta_df = self.get_data_object(name_to_delta).compute_delta(name=result_name, filter_obj=filter_obj)

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
            self.get_df(interval_index_name), mean_level=mean_level, new_level_as='frame')

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
            result_name = name_to_fit+'_fit'
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
                    self.pandas_index_manager.rename_index_level(df_fit, 'filter', 'frame')
                    if level == 'ant':
                        self.add_new2d_from_df(
                            df=df_fit, name=result_name, xname=xname, yname=yname, object_type='Events2d',
                            category=category, label=label, xlabel=xlabel, ylabel=ylabel, description=description,
                            replace=replace)
                        return result_name
                    else:
                        return df_fit
