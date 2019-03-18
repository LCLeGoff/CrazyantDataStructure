class DataSetWriter:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def write(self, dataset):
        add = self.root + dataset.category + '/' + dataset.name + '.csv'
        dataset.df.to_csv(add)
