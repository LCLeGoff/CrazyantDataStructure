from DataStructure.DataObjects.Definitions import Definitions
from Tools.MiscellaneousTools.JsonFiles import import_obj


class DefinitionLoader:
    def __init__(self, root, group):
        self.definition_dict = import_obj(root + group + '/definition_dict.json')

    def build(self, name):
        return Definitions(name, self.definition_dict[name])
