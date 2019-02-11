from Tools.MiscellaneousTools.PickleJsonFiles import import_obj_json, write_obj_json


class DefinitionDeleter:
    def __init__(self, root, group):
        self.add = root + group + '/definition_dict.json'

    def delete(self, definition):
        def_dict = import_obj_json(self.add)
        def_dict.pop(definition.name)
        write_obj_json(self.add, def_dict)
