from DataStructure.DataManager.Writers.CharacteristicTimeSeriesWriter import CharacteristicTimeSeries1dWriter
from DataStructure.DataManager.Writers.AntCharacteristicsWriter import AntCharacteristics1dWriter
from DataStructure.DataManager.Writers.CharacteristicsWriter import Characteristics1dWriter, Characteristics2dWriter
from DataStructure.DataManager.Writers.DataSetWriter import DataSetWriter
from DataStructure.DataManager.Writers.DefinitionWriter import DefinitionWriter
from DataStructure.DataManager.Writers.EventsWriter import Events1dWriter, Events2dWriter
from DataStructure.DataManager.Writers.TimeSeriesWriter import TimeSeriesWriter
from DataStructure.VariableNames import dataset_name


class DataWriter:
    def __init__(self, root, group):
        self.definition_writer = DefinitionWriter(root, group)
        self.timeseries1d_writer = TimeSeriesWriter(root, group)

        self.events1d_writer = Events1dWriter(root, group)
        self.events2d_writer = Events2dWriter(root, group)

        self.characteristics1d_writer = Characteristics1dWriter(root, group)
        self.characteristics2d_writer = Characteristics2dWriter(root, group)

        self.ant_characteristics1d_writer = AntCharacteristics1dWriter(root, group)
        self.characteristic_timeseries1d_writer = CharacteristicTimeSeries1dWriter(root, group)

        self.dataset_writer = DataSetWriter(root, group)

    def write(self, obj):

        if obj.object_type == 'TimeSeries1d':
            self.timeseries1d_writer.write(obj)

        elif obj.object_type in ['Events1d', 'CharacteristicEvents1d']:
            self.events1d_writer.write(obj)
        elif obj.object_type in ['Events2d', 'CharacteristicEvents2d']:
            self.events2d_writer.write(obj)

        elif obj.object_type == 'Characteristics1d':
            self.characteristics1d_writer.write(obj)
        elif obj.object_type == 'Characteristics2d':
            self.characteristics2d_writer.write(obj)

        elif obj.object_type == 'AntCharacteristics1d':
            self.ant_characteristics1d_writer.write(obj)

        elif obj.object_type == 'CharacteristicTimeSeries1d':
            self.characteristic_timeseries1d_writer.write(obj)

        elif obj.object_type == dataset_name:
            self.dataset_writer.write(obj)
        else:
            raise ValueError(obj.name + ' has no defined object type')
        self.definition_writer.write(obj.definition)
