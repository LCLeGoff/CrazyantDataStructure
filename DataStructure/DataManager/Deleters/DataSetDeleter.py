import os


class DataSetDeleter:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def delete(self, dataset):
        address = self.root + dataset.category + '/' + dataset.name + '.csv'
        os.remove(address)
