import os


class AntCharacteristics1dRenamer:
    def __init__(self, root, group):
        self.root = root + group + '/'

    def rename(self, ant_chara1d, name, category=None, label=None, description=None):

        old_address = self.root + ant_chara1d.category + '/' + ant_chara1d.name + '.csv'
        os.remove(old_address)

        ant_chara1d.rename(name=name, category=category, label=label, description=description)

        new_address = self.root + ant_chara1d.category + '/' + name + '.csv'
        ant_chara1d.df.to_csv(new_address)
