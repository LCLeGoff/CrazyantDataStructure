from Tools.JsonFiles import import_obj, write_obj


class DefinitionDeleter:
    def __init__(self, root, group):
        self.add = root + group + '/definition_dict.json'

    def delete(self, definition):
        def_dict = import_obj(self.add)
        def_dict.pop(definition.name)
        write_obj(self.add, def_dict)
