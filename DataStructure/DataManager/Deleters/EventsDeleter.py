import os


class Events1dDeleter:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def delete(self, events):
        if events.category == 'Raw':
            raise OSError('not allowed to delete Events of the category Raw')
        else:
            address = self.root + events.category + '/' + events.name + '.csv'
            os.remove(address)


class Events2dDeleter:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def delete(self, events):
        if events.category == 'Raw':
            raise OSError('not allowed to delete Events2d of the category Raw')
        else:
            address = self.root + events.category + '/' + events.name + '.csv'
            os.remove(address)
