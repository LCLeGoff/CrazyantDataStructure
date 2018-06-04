class Definitions:
    def __init__(self, name, attr_dict):
        self.name = name
        self.dict = attr_dict
        for key, value in attr_dict.items():
            self.__dict__[key] = value


class DefinitionBuilder:

    def __init__(self):
        pass

    @staticmethod
    def create_dict(object_type, category=None, label=None, description=None, xlabel=None, ylabel=None):
        def_dict = dict()
        def_dict['category'] = category
        def_dict['description'] = description
        def_dict['label'] = label
        def_dict['object_type'] = object_type
        if xlabel is not None and ylabel is not None:
            def_dict['xlabel'] = xlabel
            def_dict['ylabel'] = ylabel
        return def_dict

    @staticmethod
    def from_dict(name, def_dict):
        return Definitions(name, def_dict)

    @staticmethod
    def build_from_definition(class_self, definition):
        class_self.name = definition.name
        class_self.definition = definition
        for key, value in definition.dict.items():
            class_self.__dict__[key] = value

    def build1d(self, name, object_type, category=None, label=None, description=None):
        def_dict = self.create_dict(
            category=category, object_type=object_type,
            label=label, description=description
        )
        return Definitions(name, def_dict)

    def build2d(self, name, object_type, category=None, label=None, xlabel=None, ylabel=None, description=None):
        def_dict = self.create_dict(
            category=category, object_type=object_type,
            label=label, description=description,
            xlabel=xlabel, ylabel=ylabel
        )
        return Definitions(name, def_dict)
