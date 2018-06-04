from Tools.JsonFiles import import_obj
from DataObjects.Definitions import Definitions


class DefinitionLoader:
    def __init__(self, root, group):
        self.definition_dict = import_obj(root + group + '/definition_dict.json')

    def build(self, name):
        return Definitions(name, self.definition_dict[name])
