import os


class AntCharacteristics1dDeleter:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def delete(self, ant_chara):
        if ant_chara.category == 'Raw':
            raise OSError('not allowed to delete AntCharacteristics of the category Raw')
        else:
            address = self.root + ant_chara.category + '/' + ant_chara.name + '.csv'
            os.remove(address)
