from Tools.MiscellaneousTools.PickleJsonFiles import import_obj_json, write_obj_json


class DefinitionRenamer:
    def __init__(self, root, group):
        self.add = root + group + '/definition_dict.json'

    def rename(self, definition, old_name):
        def_dict = import_obj_json(self.add)
        def_dict.pop(old_name)
        def_dict[definition.name] = definition.dict
        write_obj_json(self.add, def_dict)
