class DataSetWriter:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def write(self, dataset):
        add = self.root + dataset.category + '/DataSets/' + dataset.name + '.csv'
        dataset.df.to_csv(add)
