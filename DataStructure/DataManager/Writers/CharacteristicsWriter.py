import numpy as np

from Tools.MiscellaneousTools.PickleJsonFiles import import_obj_json, write_obj_json


class Characteristics1dWriter:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def write(self, chara1d):
        address = self.root + chara1d.definition.category + '/Characteristics.json'
        chara_dict = import_obj_json(address)

        id_exp_list = chara1d.get_index_array_of_id_exp()
        for id_exp in id_exp_list:
            val = chara1d.df.loc[id_exp][chara1d.name]

            if isinstance(val, np.int64):
                val = int(val)

            chara_dict[str(id_exp)][chara1d.name] = val

        write_obj_json(address, chara_dict)


class Characteristics2dWriter:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def write(self, chara2d):
        address = self.root + chara2d.definition.category + '/Characteristics.json'
        chara_dict = import_obj_json(address)

        id_exp_list = chara2d.get_index_array_of_id_exp()
        for id_exp in id_exp_list:
            val0 = chara2d.df.loc[id_exp][chara2d.df.columns[0]]
            val1 = chara2d.df.loc[id_exp][chara2d.df.columns[1]]

            if isinstance(val0, np.int64):
                val0 = int(val0)
            if isinstance(val1, np.int64):
                val1 = int(val1)

            chara_dict[str(id_exp)][chara2d.name] = [val0, val1]

        write_obj_json(address, chara_dict)
