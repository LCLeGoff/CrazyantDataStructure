import os
import pandas as pd

from DataStructure.DataManager.Deleters.DataDeleter import DataDeleter
from DataStructure.DataManager.Loaders.DataLoader import DataLoader
from DataStructure.DataManager.Renamers.DataRenamer import DataRenamer
from DataStructure.DataManager.Writers.DataWriter import DataWriter
from DataStructure.VariableNames import dataset_name
from Tools.MiscellaneousTools.PickleJsonFiles import import_id_exp_list, write_obj_json


class DataFileManager:

    def __init__(self, root, group):
        self.root = root + group + '/'

        self.data_loader = DataLoader(root, group)
        self.data_writer = DataWriter(root, group)
        self.data_deleter = DataDeleter(root, group)
        self.data_renamer = DataRenamer(root, group)

        self.existing_categories = set([
            self.data_loader.definition_loader.definition_dict[key]['category']
            for key in self.data_loader.definition_loader.definition_dict.keys()])

        self.id_exp_list = import_id_exp_list(self.root)
        self.id_exp_list.sort()
        self.exp_ant_frame_index = None
        self.exp_frame_index = None

    def _get_exp_ant_frame_index(self):
        if self.exp_ant_frame_index is None:
            self.data_loader.timeseries1d_loader.load_category('Raw')
            self.exp_ant_frame_index = self.data_loader.timeseries1d_loader.categories['Raw'].index
        return self.exp_ant_frame_index

    def _get_exp_frame_index(self):
        if self.exp_frame_index is None:
            self.data_loader.characteristic_timeseries1d_loader.load_category('Raw')
            self.exp_frame_index = self.data_loader.characteristic_timeseries1d_loader.categories['Raw'].index
        return self.exp_frame_index

    def load(self, name):
        if self.is_name_in_data(name) == 1:
            return self.data_loader.load(name)
        else:
            raise NameError(name + ' does not exist')

    def create_new_category(self, category):

        if category is not None:
            add = self.root + category + '/'

            if not (os.path.isdir(add)):
                print('Create category '+category)

                try:
                    os.mkdir(add)
                    os.mkdir(add+'Plots/')
                    os.mkdir(add+'DataSets/')
                except FileExistsError:
                    pass

                chara = dict()
                for id_exp in self.id_exp_list:
                    chara[str(id_exp)] = dict()

                write_obj_json(add + 'Characteristics.json', chara)

                df = pd.DataFrame(index=self._get_exp_ant_frame_index())
                df.to_csv(add + 'TimeSeries.csv')

                df = pd.DataFrame(index=self._get_exp_frame_index())
                df.to_csv(add + 'CharacteristicTimeSeries.csv')

    def write(self, obj):
        if obj.category is None:
            raise ValueError(obj.name + ' definition not properly set: category is missing')
        elif obj.label is None:
            raise ValueError(obj.name + ' definition not properly set: label is missing')
        elif obj.description is None:
            raise ValueError(obj.name + ' definition not properly set: description is missing')
        else:
            if obj.data_type == dataset_name and 'nb_indexes' not in obj.definition:
                raise ValueError(obj.name + ' definition not properly set: nb_indexes is missing')
            else:
                self.create_new_category(obj.category)
                self.data_writer.write(obj)

    def rename(
            self, obj, name=None, xname=None, yname=None,
            category=None, label=None, xlabel=None, ylabel=None, description=None):

        self.create_new_category(category)
        self.data_renamer.rename(
            obj, name=name, xname=xname, yname=yname, category=category,
            label=label, xlabel=xlabel, ylabel=ylabel, description=description)

    def delete(self, obj):
        self.data_deleter.delete(obj)

    def is_name_in_data(self, name):
        return int(name in self.data_loader.definition_loader.definition_dict.keys())
