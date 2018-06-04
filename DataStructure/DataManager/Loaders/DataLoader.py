from DataStructure.DataManager.Loaders.AntCharacteristicsLoader import AntCharacteristics1dLoader
from DataStructure.DataManager.Loaders.CharacteristicsLoader import Characteristics1dLoader, Characteristics2dLoader
from DataStructure.DataManager.Loaders.DefinitionLoader import DefinitionLoader
from DataStructure.DataManager.Loaders.EventsLoader import Events1dLoader, Events2dLoader
from DataStructure.DataManager.Loaders.TimeSeriesLoader import TimeSeries1dLoader


class DataLoader:
    def __init__(self, root, group):
        self.root = root + group + '/'
        self.definition_loader = DefinitionLoader(root, group)
        self.time_series_loader = TimeSeries1dLoader(root, group)
        self.events_loader = Events1dLoader(root, group)
        self.events2d_loader = Events2dLoader(root, group)
        self.characteristics1d_loader = Characteristics1dLoader(root, group)
        self.characteristics2d_loader = Characteristics2dLoader(root, group)
        self.ant_characteristics1d_loader = AntCharacteristics1dLoader(root, group)

    def load(self, name):
        definition = self.definition_loader.build(name)
        if definition.object_type == 'TimeSeries1d':
            res = self.time_series_loader.load(definition)
        elif definition.object_type == 'Events1d':
            res = self.events_loader.load(definition)
        elif definition.object_type == 'Events2d':
            res = self.events2d_loader.load(definition)
        elif definition.object_type == 'Characteristics1d':
            res = self.characteristics1d_loader.load(definition)
        elif definition.object_type == 'Characteristics2d':
            res = self.characteristics2d_loader.load(definition)
        elif definition.object_type == 'AntCharacteristics1d':
            res = self.ant_characteristics1d_loader.load(definition)
        else:
            raise ValueError(name + ' has no defined object type :')
        return res
