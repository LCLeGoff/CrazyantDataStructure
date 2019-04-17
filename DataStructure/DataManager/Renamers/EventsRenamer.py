import os


class Events1dRenamer:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def rename(self, event1d, name, category=None, label=None, description=None):

        old_address = self.root + event1d.category + '/DataSets/' + event1d.name + '.csv'
        os.remove(old_address)

        event1d.rename(name=name, category=category, label=label, description=description)

        new_address = self.root + event1d.category + '/DataSets/' + name + '.csv'
        event1d.df.to_csv(new_address)


class Events2dRenamer:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def rename(
            self, event2d, name, xname=None, yname=None, category=None,
            label=None, xlabel=None, ylabel=None, description=None):

        old_address = self.root + event2d.category + '/DataSets/' + event2d.name + '.csv'
        os.remove(old_address)

        event2d.rename(
            name=name, xname=xname, yname=yname, category=category,
            label=label, xlabel=xlabel, ylabel=ylabel, description=description)

        new_address = self.root + event2d.category + '/DataSets/' + name + '.csv'
        event2d.df.to_csv(new_address)
