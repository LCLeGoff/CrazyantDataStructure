from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from ExperimentGroups import ExperimentGroups


class AnalyseFoodCarrying:

    def __init__(self, root, group, exp: ExperimentGroups = None):
        if exp is None:
            self.exp = ExperimentGroupBuilder(root).build(group)
        else:
            self.exp = exp

    def compute_carrying_intervals(self):

        name = 'carrying'
        self.exp.load([name+'_next2food', 'x'])

        self.exp.add_copy(
            old_name='x', new_name=name, category='FoodCarrying',
            label='Is ant carrying?', description='Boolean giving if ants are carrying or not'
        )

        df = self.exp.get_reindexed_df(name_to_reindex=name+'_next2food', reindexer_name='x', fill_value=0)

        self.exp.__dict__[name].df = df

        result_name = self.exp.compute_time_intervals(
            name_to_intervals=name, category='FoodCarrying',
            label='Carrying time intervals', description='Time intervals during which ants are carrying (in frame)'
        )

        self.exp.write(result_name)
