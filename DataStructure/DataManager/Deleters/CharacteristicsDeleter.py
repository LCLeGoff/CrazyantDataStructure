from Tools.MiscellaneousTools.PickleJsonFiles import import_obj_json, write_obj_json


class Characteristics1dDeleter:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def delete(self, chara1d):
        if chara1d.category == 'Raw':
            raise OSError('not allowed to modify Characteristics of the category Raw')
        else:
            address = self.root + chara1d.definition.category + '/Characteristics.json'
            chara_dict = import_obj_json(address)

            id_exp_list = chara1d.get_index_array_of_id_exp()
            for id_exp in id_exp_list:
                chara_dict[str(id_exp)].pop(chara1d.name)

            write_obj_json(address, chara_dict)
