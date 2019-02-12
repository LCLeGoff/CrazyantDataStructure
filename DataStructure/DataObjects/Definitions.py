class Definitions:
    def __init__(self, name, attr_dict):
        self.name = name
        self.dict = attr_dict
        for key, value in attr_dict.items():
            self.__dict__[key] = value

    def rename(self, name=None, category=None, label=None, xlabel=None, ylabel=None, description=None):
        self.change_name(name)
        self.change_category(category)
        self.change_label(label)
        self.change_description(description)
        self.change_xlabel(xlabel)
        self.change_ylabel(ylabel)

    def change_name(self, name):
        if name is not None:
            self.name = name

    def change_category(self, category):
        if category is not None:
            self.dict['category'] = category

    def change_label(self, label):
        if label is not None:
            self.dict['label'] = label

    def change_description(self, description):
        if description is not None:
            self.dict['description'] = description

    def change_xlabel(self, xlabel):
        if xlabel is not None and 'xlabel' in self.dict:
            self.dict['xlabel'] = xlabel

    def change_ylabel(self, ylabel):
        if ylabel is not None and 'ylabel' in self.dict:
            self.dict['ylabel'] = ylabel


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
    def add_definition_to_class(class_self, definition):
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
