from DataStructure.DataObjects.AntCharacteristics1d import AntCharacteristics1dBuilder
from DataStructure.DataObjects.CharacteristicEvents1d import CharacteristicEvents1dBuilder
from DataStructure.DataObjects.CharacteristicTimeSeries1d import CharacteristicTimeSeries1dBuilder
from DataStructure.DataObjects.Characteristics1d import Characteristics1dBuilder
from DataStructure.DataObjects.Characteristics2d import Characteristics2dBuilder
from DataStructure.DataObjects.DataSet import DataSetBuilder
from DataStructure.DataObjects.Events1d import Events1dBuilder
from DataStructure.DataObjects.Events2d import Events2dBuilder
from DataStructure.DataObjects.TimeSeries1d import TimeSeries1dBuilder
from DataStructure.DataObjects.TimeSeries2d import TimeSeries2dBuilder
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name, dataset_name
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class Builder:
    def __init__(self):
        self.pandas_index_manager = PandasIndexManager()

    @staticmethod
    def build_dataset_from_df(df, name, category=None, label=None, description=None):
        return DataSetBuilder.build(df=df, name=name, category=category, label=label,
                                    description=description, nb_indexes=len(df.index.names))

    @staticmethod
    def build_dataset_from_array(array, name, index_names, column_names, category=None, label=None, description=None):

        if len(array) == 0:
            df = Builder().pandas_index_manager.create_empty_df(index_names=index_names, column_names=column_names)

        else:
            df = Builder().pandas_index_manager.convert_array_to_df(array=array, index_names=index_names,
                                                                    column_names=column_names)

        return Builder.build_dataset_from_df(df=df, name=name, category=category, label=label, description=description)

    @staticmethod
    def build1d_from_df(df, name, object_type, category=None, label=None, description=None):
        if object_type == 'TimeSeries1d':
            return TimeSeries1dBuilder.build(df=df, name=name, category=category, label=label, description=description)

        elif object_type == 'Events1d':
            return Events1dBuilder.build(df=df, name=name, category=category, label=label, description=description)

        elif object_type == 'Characteristics1d':
            return Characteristics1dBuilder.build(df=df, name=name, category=category,
                                                  label=label, description=description)
        elif object_type == 'AntCharacteristics1d':
            return AntCharacteristics1dBuilder.build(df=df, name=name, category=category,
                                                     label=label, description=description)
        elif object_type == 'CharacteristicTimeSeries1d':
            return CharacteristicTimeSeries1dBuilder.build(df=df, name=name, category=category,
                                                           label=label, description=description)
        elif object_type == 'CharacteristicEvents1d':
            return CharacteristicEvents1dBuilder.build(df=df, name=name, category=category,
                                                       label=label, description=description)
        elif object_type == dataset_name:
            raise TypeError('Use build_dataset_from_df for DataSets')

        else:
            raise TypeError('Type ' + object_type + ' is unknown or 2d')

    @staticmethod
    def build2d_from_df(
            df, name, xname, yname, object_type, category=None, label=None, xlabel=None, ylabel=None, description=None):

        if object_type == 'TimeSeries2d':
            return TimeSeries2dBuilder.build_from_df(
                df=df.sort_index(), name=name, xname=xname, yname=yname,
                category=category, label=label, xlabel=xlabel, ylabel=ylabel,
                description=description)

        elif object_type == 'Events2d':
            return Events2dBuilder.build_from_df(
                df=df.sort_index(), name=name, xname=xname, yname=yname,
                category=category, label=label, xlabel=xlabel, ylabel=ylabel,
                description=description)

        elif object_type == 'Characteristics2d':
            return Characteristics2dBuilder.build_from_df(
                df=df.sort_index(), name=name, xname=xname, yname=yname,
                category=category, label=label, xlabel=xlabel, ylabel=ylabel,
                description=description)

        elif object_type == dataset_name:
            raise TypeError('Use build_dataset_from_df for DataSets')

        else:
            raise TypeError('Type ' + object_type + ' is unknown or 1d')

    @staticmethod
    def build2d_from_array(
            array, name, xname, yname, object_type, category=None, label=None, xlabel=None, ylabel=None,
            description=None):

        if object_type == dataset_name:
            raise TypeError('Use build_dataset_from_array for DataSets')

        else:
            if len(array) == 0:
                df = Builder().pandas_index_manager.create_empty_df(
                    index_names=[id_exp_name, id_ant_name, id_frame_name], column_names=[xname, yname])
            else:
                df = Builder().pandas_index_manager.convert_array_to_df(
                    array=array, index_names=[id_exp_name, id_ant_name, id_frame_name], column_names=[xname, yname])

            return Builder.build2d_from_df(
                df=df, object_type=object_type,
                name=name, xname=xname, yname=yname,
                category=category, label=label, xlabel=xlabel, ylabel=ylabel,
                description=description)
