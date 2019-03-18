from DataStructure.DataManager.Deleters.CharacteristicTimeSeriesDeleter import CharacteristicTimeSeries1dDeleter
from DataStructure.DataManager.Deleters.AntCharacteristicsDeleter import AntCharacteristics1dDeleter
from DataStructure.DataManager.Deleters.CharacteristicsDeleter import Characteristics1dDeleter
from DataStructure.DataManager.Deleters.DataSetDeleter import DataSetDeleter
from DataStructure.DataManager.Deleters.DefinitionDeleter import DefinitionDeleter
from DataStructure.DataManager.Deleters.EventsDeleter import Events1dDeleter, Events2dDeleter
from DataStructure.DataManager.Deleters.TimeSeriesDeleter import TimeSeriesDeleter
from DataStructure.VariableNames import dataset_name


class DataDeleter:
    def __init__(self, root, group):
        self.definition_deleter = DefinitionDeleter(root, group)
        self.timeseries1d_deleter = TimeSeriesDeleter(root, group)
        self.events1d_deleter = Events1dDeleter(root, group)
        self.events2d_deleter = Events2dDeleter(root, group)
        self.characteristics1d_deleter = Characteristics1dDeleter(root, group)
        self.ant_characteristics1d_deleter = AntCharacteristics1dDeleter(root, group)
        self.characteristic_timeseries1d_deleter = CharacteristicTimeSeries1dDeleter(root, group)
        self.dataset_deleter = DataSetDeleter(root, group)

    def delete(self, obj):
        if obj.object_type == 'TimeSeries1d':
            self.timeseries1d_deleter.delete(obj)
        elif obj.object_type == 'Events1d':
            self.events1d_deleter.delete(obj)
        elif obj.object_type == 'Events2d':
            self.events2d_deleter.delete(obj)
        elif obj.object_type == 'Characteristics1d':
            self.characteristics1d_deleter.delete(obj)
        elif obj.object_type == 'AntCharacteristics1d':
            self.ant_characteristics1d_deleter.delete(obj)
        elif obj.object_type == 'CharacteristicTimeSeries1d':
            self.characteristic_timeseries1d_deleter.delete(obj)
        elif obj.object_type == dataset_name:
            self.dataset_deleter.delete(obj)
        else:
            raise ValueError(obj.name + ' has no defined object type')
        self.definition_deleter.delete(obj.definition)
