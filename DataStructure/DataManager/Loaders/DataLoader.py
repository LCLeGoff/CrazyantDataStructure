from DataStructure.DataManager.Loaders.CharacteristicEventsLoader import CharacteristicEvents2dLoader, \
    CharacteristicEvents1dLoader
from DataStructure.DataManager.Loaders.CharacteristicTimeSeriesLoader import CharacteristicTimeSeries1dLoader
from DataStructure.DataManager.Loaders.AntCharacteristicsLoader import AntCharacteristics1dLoader
from DataStructure.DataManager.Loaders.CharacteristicsLoader import Characteristics1dLoader, Characteristics2dLoader
from DataStructure.DataManager.Loaders.DataSetLoader import DataSetLoader
from DataStructure.DataManager.Loaders.DefinitionLoader import DefinitionLoader
from DataStructure.DataManager.Loaders.EventsLoader import Events1dLoader, Events2dLoader
from DataStructure.DataManager.Loaders.TimeSeriesLoader import TimeSeries1dLoader
from DataStructure.VariableNames import dataset_name


class DataLoader:
    def __init__(self, root, group):
        self.root = root + group + '/'

        self.definition_loader = DefinitionLoader(root, group)

        self.timeseries1d_loader = TimeSeries1dLoader(root, group)

        self.events1d_loader = Events1dLoader(root, group)
        self.events2d_loader = Events2dLoader(root, group)

        self.characteristic_events1d_loader = CharacteristicEvents1dLoader(root, group)
        self.characteristic_events2d_loader = CharacteristicEvents2dLoader(root, group)

        self.characteristics1d_loader = Characteristics1dLoader(root, group)
        self.characteristics2d_loader = Characteristics2dLoader(root, group)

        self.ant_characteristics1d_loader = AntCharacteristics1dLoader(root, group)

        self.characteristic_timeseries1d_loader = CharacteristicTimeSeries1dLoader(root, group)

        self.dataset_loader = DataSetLoader(root, group)

    def load(self, name):
        definition = self.definition_loader.build(name)

        if definition.object_type == 'TimeSeries1d':
            res = self.timeseries1d_loader.load(definition)

        elif definition.object_type == 'Events1d':
            res = self.events1d_loader.load(definition)
        elif definition.object_type == 'Events2d':
            res = self.events2d_loader.load(definition)

        elif definition.object_type == 'CharacteristicEvents1d':
            res = self.characteristic_events1d_loader.load(definition)
        elif definition.object_type == 'CharacteristicEvents2d':
            res = self.characteristic_events2d_loader.load(definition)

        elif definition.object_type == 'Characteristics1d':
            res = self.characteristics1d_loader.load(definition)
        elif definition.object_type == 'Characteristics2d':
            res = self.characteristics2d_loader.load(definition)

        elif definition.object_type == 'AntCharacteristics1d':
            res = self.ant_characteristics1d_loader.load(definition)

        elif definition.object_type == 'CharacteristicTimeSeries1d':
            res = self.characteristic_timeseries1d_loader.load(definition)

        elif definition.object_type == dataset_name:
            res = self.dataset_loader.load(definition)

        else:
            raise ValueError(name + ' has no defined object type :')
        return res
