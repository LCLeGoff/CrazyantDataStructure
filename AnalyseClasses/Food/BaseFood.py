import numpy as np

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Tools.MiscellaneousTools.Geometry import distance
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class AnalyseBaseFood:
    def __init__(self, root, group):
        self.pd_idx_manager = PandasIndexManager()
        self.exp = ExperimentGroupBuilder(root).build(group)

    def compute_traj_next_food(self):
        name = 'food_distance'
        print(name)
        self.exp.load(['food_x', 'food_y', 'x', 'y'])

        list_id = self.exp.x.df.index.get_values()
        d = np.zeros((len(self.exp.x.df), 4))*np.nan
        for (id_exp, frame) in self.exp.food_x.df.index:
            print(id_exp, frame)
            x = self.exp.x.df.loc[id_exp, :, frame]
            y = self.exp.y.df.loc[id_exp, :, frame]

            food_x = float(self.exp.food_x.df.loc[id_exp, frame])
            food_y = float(self.exp.food_y.df.loc[id_exp, frame])

            x = (x-food_x)**2
            y = (y-food_y)**2

            d = np.round(np.sqrt(x.x+y.y), 3)

        # d = np.zeros((len(self.exp.x.df), 4))*np.nan
        # for (id_exp, frame) in self.exp.food_x.df.index:
        #     print(id_exp, frame)
        #     x = self.exp.x.df.loc[id_exp, :, frame]
        #     y = self.exp.y.df.loc[id_exp, :, frame]
        #
        #     food_x = float(self.exp.food_x.df.loc[id_exp, frame])
        #     food_y = float(self.exp.food_y.df.loc[id_exp, frame])
        #
        #     x = (x-food_x)**2
        #     y = (y-food_y)**2
        #
        #     d = np.round(np.sqrt(x.x+y.y), 3)

        # if (id_exp, frame) in self.exp.food_x.df.index:
        # for i, (id_exp, id_ant, frame) in enumerate(list_id_exp_ant_frame):
        #     print(id_exp, id_ant, frame)
        #     if (id_exp, frame) in self.exp.food_x.df.index:
        #         x = self.exp.x.df.loc[id_exp, id_ant, frame]
        #         y = self.exp.y.df.loc[id_exp, id_ant, frame]
        #
        #         x_food = self.exp.food_x.df.loc[id_exp, frame]
        #         y_food = self.exp.food_y.df.loc[id_exp, frame]
        #
        #         d[i, -1] = distance([x, y], [x_food, y_food])[0][0]

        self.exp.add_new1d_from_array(
            array=d, name=name, object_type='TimeSeries1d',
            category='FoodBase', label='Food distance', description='Distance between the food and the ants'
        )

        self.exp.write(name)
