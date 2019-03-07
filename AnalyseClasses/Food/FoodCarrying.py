import matplotlib.pyplot as plt
import numpy as np

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from ExperimentGroups import ExperimentGroups


class AnalyseFoodCarrying:

    def __init__(self, root, group, exp: ExperimentGroups = None):
        if exp is None:
            self.exp = ExperimentGroupBuilder(root).build(group)
        else:
            self.exp = exp

    def compute_carrying_intervals(self, redo=False):

        result_name = 'carrying_intervals'

        if redo is True:
            name = 'carrying'
            self.exp.load([name+'_next2food', 'x'])

            self.exp.add_copy(
                old_name='x', new_name=name, category='FoodCarrying',
                label='Is ant carrying?', description='Boolean giving if ants are carrying or not'
            )

            df = self.exp.get_reindexed_df(name_to_reindex=name+'_next2food', reindexer_name='x', fill_value=0)

            self.exp.get_data_object(name).df = df

            self.exp.compute_time_intervals(
                name_to_intervals=name, category='FoodCarrying', result_name=result_name,
                label='Carrying time intervals', description='Time intervals during which ants are carrying (in frame)'
            )

            self.exp.write(result_name)

        else:
            self.exp.load(result_name)

        self.exp.operation(result_name, lambda x: x/100.)
        self.exp.get_data_object(result_name).plotter.hist1d(
            bins=np.arange(0, 1e2, 1/100.), marker='o', ls='', xscale='log', yscale='log')
        # plt.show()

    def compute_not_carrying_intervals(self, redo=False):

        result_name = 'not_carrying_intervals'

        if redo is True:
            name = 'carrying'
            self.exp.load([name+'_next2food', 'x'])

            self.exp.add_copy(
                old_name='x', new_name=name, category='FoodCarrying',
                label='Is ant carrying?', description='Boolean giving if ants are carrying or not'
            )

            df = self.exp.get_reindexed_df(name_to_reindex=name+'_next2food', reindexer_name='x', fill_value=0)

            self.exp.__dict__[name].df = 1-df

            self.exp.compute_time_intervals(
                name_to_intervals=name, result_name=result_name,
                category='FoodCarrying', label='Not carrying time intervals',
                description='Time intervals during which ants are not carrying (in frame)'
            )

            self.exp.write(result_name)

        else:
            self.exp.load(result_name)

        self.exp.operation(result_name, lambda x: x/100.)
        self.exp.get_data_object(result_name).plotter.hist1d(marker='o', ls='', xscale='log', yscale='log')
        plt.show()
