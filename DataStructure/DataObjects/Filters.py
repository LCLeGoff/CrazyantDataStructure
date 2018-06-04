from DataStructure.DataObjects.Events1d import Events1dBuilder


class Filters:
    def __init__(self):
        pass

    @staticmethod
    def filter(obj, event, name, label=None, category=None, description=None):
        index = event.df.index
        df = obj.df.loc[index, :]
        return Events1dBuilder.build(
            df=df, name=name, category=category, label=label, description=description)
