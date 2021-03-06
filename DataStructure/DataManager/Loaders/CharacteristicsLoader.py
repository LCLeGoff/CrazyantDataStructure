import pandas as pd
import numpy as np

from DataStructure.DataObjects.Characteristics1d import Characteristics1d
from DataStructure.DataObjects.Characteristics2d import Characteristics2d
from DataStructure.VariableNames import id_exp_name
from Tools.MiscellaneousTools.PickleJsonFiles import import_obj_json


class Characteristics1dLoader:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def load(self, definition):
        chara_dict = import_obj_json(self.root + definition.category + '/Characteristics.json')
        res = pd.DataFrame(
            [chara_dict[key][definition.name] for key in chara_dict.keys()],
            index=np.array(list(chara_dict.keys()), dtype=int),
            columns=[definition.name])
        res.index.name = id_exp_name
        res.sort_index(inplace=True)
        return Characteristics1d(res, definition)


class Characteristics2dLoader:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def load(self, definition):
        chara_dict = import_obj_json(self.root + definition.category + '/Characteristics.json')
        res = pd.DataFrame(
            [np.array(chara_dict[key][definition.name], dtype=float) for key in chara_dict.keys()],
            index=np.array(list(chara_dict.keys()), dtype=int),
            columns=['x', 'y'])
        res.index.name = id_exp_name
        res.sort_index(inplace=True)
        return Characteristics2d(res, definition)
