from Tools.MiscellaneousTools.PickleJsonFiles import import_obj_json, write_obj_json


class DefinitionWriter:
    def __init__(self, root, group):
        self.add = root + group + '/definition_dict.json'

    def write(self, definition):
        def_dict = import_obj_json(self.add)
        def_dict[definition.name] = definition.dict
        write_obj_json(self.add, def_dict)
