import os


class DataSetRenamer:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def rename(self, dataset, name, category=None, label=None, description=None):

        old_address = self.root + dataset.category + '/DataSets/' + dataset.name + '.csv'

        dataset.rename(name=name, category=category, label=label, description=description)

        new_address = self.root + dataset.category + '/DataSets/' + name + '.csv'
        dataset.df.to_csv(new_address)

        os.remove(old_address)
