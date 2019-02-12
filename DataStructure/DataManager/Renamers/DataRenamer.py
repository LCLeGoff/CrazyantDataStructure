from DataStructure.DataManager.Renamers.AntCharacteristicsRenamer import AntCharacteristics1dRenamer
from DataStructure.DataManager.Renamers.CharacteristicTimeSeriesRenamer import CharacteristicTimeSeries1dRenamer
from DataStructure.DataManager.Renamers.CharacteristicsRenamer import Characteristics1dRenamer
from DataStructure.DataManager.Renamers.DefinitionRenamer import DefinitionRenamer
from DataStructure.DataManager.Renamers.EventsRenamer import Events1dRenamer, Events2dRenamer
from DataStructure.DataManager.Renamers.TimeSeriesRenamer import TimeSeries1dRenamer


class DataRenamer:
    def __init__(self, root, group):
        self.definition_renamer = DefinitionRenamer(root, group)
        self.timeseries1d_renamer = TimeSeries1dRenamer(root, group)
        self.events1d_renamer = Events1dRenamer(root, group)
        self.events2d_renamer = Events2dRenamer(root, group)
        self.characteristics1d_renamer = Characteristics1dRenamer(root, group)
        self.ant_characteristics1d_renamer = AntCharacteristics1dRenamer(root, group)
        self.characteristic_timeseries1d_renamer = CharacteristicTimeSeries1dRenamer(root, group)

    def rename(
            self, obj, name, xname=None, yname=None, category=None,
            label=None, description=None, xlabel=None, ylabel=None):

        old_name = obj.name

        if obj.object_type == 'TimeSeries1d':
            self.timeseries1d_renamer.rename(
                ts=obj, name=name, category=category, label=label, description=description)

        elif obj.object_type == 'Events1d':
            self.events1d_renamer.rename(
                event1d=obj, name=name, category=category, label=label, description=description)

        elif obj.object_type == 'Events2d':
            self.events2d_renamer.rename(
                event2d=obj, name=name, xname=xname, yname=yname, category=category,
                label=label, xlabel=xlabel, ylabel=ylabel, description=description)

        elif obj.object_type == 'Characteristics1d':
            self.characteristics1d_renamer.rename(
                chara1d=obj, name=name, category=category, label=label, description=description)

        elif obj.object_type == 'AntCharacteristics1d':
            self.ant_characteristics1d_renamer.rename(
                ant_chara1d=obj, name=name, category=category, label=label, description=description)

        elif obj.object_type == 'CharacteristicTimeSeries1d':
            self.characteristic_timeseries1d_renamer.rename(
                chara_ts1d=obj, name=name, category=category, label=label, description=description)

        else:
            raise ValueError(obj.name + ' has no defined object type')
        self.definition_renamer.rename(obj.definition, old_name)
