import pandas as pd

from DataStructure.DataObjects.CharacteristicTimeSeries1d import CharacteristicTimeSeries1d


class CharacteristicTimeSeries1dLoader:
    def __init__(self, root, group):
        self.root = root + group + '/'
        self.categories = dict()

    def load_category(self, category):
        add = self.root + category + '/CharacteristicTimeSeries.csv'
        if not (category in self.categories.keys()):
            self.categories[category] = pd.read_csv(add, index_col=['id_exp', 'frame'])

    def load(self, definition):
        self.load_category(definition.category)
        return CharacteristicTimeSeries1d(
            pd.DataFrame(self.categories[definition.category][definition.name]), definition)
