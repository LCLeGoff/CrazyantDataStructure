import json
import pickle
import numpy as np


# ToDo: put it somewhere elseFalse
def import_id_exp_list(root):
    return np.array(list(import_obj_json(root + 'Raw/Characteristics.json').keys()), dtype=int)


def write_obj_json(add, obj, rw='w'):
    out_file = open(add, rw)
    json.dump(obj, out_file, indent=2)
    out_file.close()


def write_obj_pickle(add, obj, rw='wb'):
    out_file = open(add, rw)
    pickle.dump(obj, out_file)
    out_file.close()


def import_obj_json(add):
    in_file = open(add, 'r')
    new_dict = json.load(in_file)
    in_file.close()
    return new_dict


def import_obj_pickle(add):
    in_file = open(add, 'rb')
    new_dict = pickle.load(in_file)
    in_file.close()
    return new_dict


def add_to_dict_json(self, add, key, values):
    res = self.import_obj(add)
    if key in res.keys():
        print(key + 'already exists')
    else:
        res[key] = values
        self.write_obj(add, res)


def remove_from_dict_json(self, add, key):
    res = self.import_obj(add)
    if key in res.keys():
        res.pop(key)
        self.write_obj(add, res)
    else:
        raise KeyError(key + 'does not exist')


def import_from_dict_json(self, add, key, arr=True):
    if arr:
        return np.array(self.import_obj(add)[key])
    else:
        return self.import_obj(add)[key]
