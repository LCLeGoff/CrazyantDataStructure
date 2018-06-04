import os
import pandas as pd

from DataStructure.DataManager.Deleters.DataDeleter import DataDeleter
from DataStructure.DataManager.Loaders.DataLoader import DataLoader
from DataStructure.DataManager.Writers.DataWriter import DataWriter
from Tools.MiscellaneousTools.JsonFiles import import_id_exp_list, write_obj


class DataFileManager:

    def __init__(self, root, group):
        self.root = root + group + '/'

        self.data_loader = DataLoader(root, group)
        self.data_writer = DataWriter(root, group)
        self.data_deleter = DataDeleter(root, group)

        self.existing_categories = set([
            self.data_loader.definition_loader.definition_dict[key]['category']
            for key in self.data_loader.definition_loader.definition_dict.keys()])

        self.id_exp_list = import_id_exp_list(self.root)
        self.id_exp_list.sort()
        self.exp_ant_frame_index = None

    def get_exp_ant_frame_index(self):
        if self.exp_ant_frame_index is None:
            self.data_loader.time_series_loader.load_category('Raw')
            self.exp_ant_frame_index = self.data_loader.time_series_loader.categories['Raw'].index
        return self.exp_ant_frame_index

    def load(self, name):
        if name in self.data_loader.definition_loader.definition_dict.keys():
            return self.data_loader.load(name)
        else:
            raise NameError(name + ' does not exist')

    def create_new_category(self, category):
        add = self.root + category + '/'
        if not (os.path.isdir(add)):
            try:
                os.mkdir(add)
            except FileExistsError:
                pass
            chara = dict()
            for id_exp in self.id_exp_list:
                chara[str(id_exp)] = dict()
            write_obj(add + 'Characteristics.json', chara)
            df = pd.DataFrame(index=self.get_exp_ant_frame_index())
            df.to_csv(add + 'TimeSeries.csv')

    def write(self, obj):
        if obj.category is None or obj.label is None or obj.description is None:
            raise ValueError(obj.name + ' definition not properly set')
        else:
            self.create_new_category(obj.category)
            self.data_writer.write(obj)

    def delete(self, obj):
        self.data_deleter.delete(obj)
