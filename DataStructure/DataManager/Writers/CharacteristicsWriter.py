import numpy as np

from Tools.MiscellaneousTools.JsonFiles import import_obj, write_obj


class Characteristics1dWriter:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def write(self, ant_chara):
        if ant_chara.category == 'Raw':
            raise OSError('not allowed to modify Characteristics of the category Raw')
        else:
            address = self.root + ant_chara.definition.category + '/Characteristics.json'
            chara_dict = import_obj(address)
            id_exp_list = ant_chara.get_index_array_of_id_exp()
            for id_exp in id_exp_list:
                val = ant_chara.df.loc[id_exp][ant_chara.name]
                if isinstance(val, float):
                    val = np.around(val, 3)
                chara_dict[str(id_exp)][ant_chara.name] = val
            write_obj(address, chara_dict)
