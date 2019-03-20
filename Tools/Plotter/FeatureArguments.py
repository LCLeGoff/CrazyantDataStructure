class ArgumentsTools:
    def __init__(self, class_self):
        self.class_self = class_self

    def add_arguments(self, attribute_name, arg_class, **kwargs):
        self.class_self.__dict__[attribute_name] = arg_class.__dict__
        self.change_arg_value(attribute_name, kwargs)

    def change_arg_value(self, attribute_name, kwargs):
        for arg_name in kwargs:
            if arg_name in self.class_self.__dict__[attribute_name]:
                self.class_self.__dict__[attribute_name][arg_name] = kwargs[arg_name]


class LineFeatureArguments:
    def __init__(self, ls='-', lw=2, c='k', alpha=1, marker='.', markeredgecolor=None, ms=8):
        self.ls = ls
        self.lw = lw
        self.c = c
        self.alpha = alpha
        self.ms = ms
        self.marker = marker
        if markeredgecolor is None:
            self.markeredgecolor = c
        else:
            self.markeredgecolor = markeredgecolor


class AxisFeatureArguments:
    def __init__(self, xscale=None, yscale=None, xlabel=None, ylabel=None):
        self.xscale = xscale
        self.yscale = yscale
        self.xlabel = xlabel
        self.ylabel = ylabel


class ArenaFeatureArguments:
    def __init__(self, arena_color='paleturquoise', arena_lw=5):
        self.c = arena_color
        self.lw = arena_lw


class FoodFeatureArguments:
    def __init__(self, food_color='paleturquoise', food_lw=5):
        self.c = food_color
        self.lw = food_lw


class GateFeatureArguments:
    def __init__(self, gate_color='k', gate_lw=10):
        self.c = gate_color
        self.lw = gate_lw
