from Tools.MiscellaneousTools.PickleJsonFiles import import_obj_json, write_obj_json


class Characteristics1dRenamer:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def rename(self, chara1d, name, category=None, label=None, description=None):
        chara1d.rename(name=name, category=category, label=label, description=description)

        address = self.root + chara1d.definition.category + '/Characteristics.json'
        chara_dict = import_obj_json(address)

        id_exp_list = chara1d.get_index_array_of_id_exp()
        for id_exp in id_exp_list:
            chara_dict[str(id_exp)][name] = chara_dict[str(id_exp)][chara1d.name]
            chara_dict[str(id_exp)].pop(chara1d.name)

        write_obj_json(address, chara_dict)
