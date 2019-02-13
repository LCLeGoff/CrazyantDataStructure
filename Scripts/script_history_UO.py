from math import pi

import matplotlib.pyplot as plt

from AnalyseClasses.AnalyseStarter import AnalyseStarter
from AnalyseClasses.Food.BaseFood import AnalyseBaseFood
from AnalyseClasses.Trajectory.BaseTrajectory import AnalyseTrajectory
from Scripts.root import root
from Tools.MiscellaneousTools.Geometry import angle

group = 'UO'

starter = AnalyseStarter(root, group)
# starter.start(redo=False, index_dicts=True, markings=False, dynamic_food=True)

traj = AnalyseTrajectory(root, group)
# traj.initialize_xy_orientation_food(dynamic_food=True)
# traj.compute_speed()

BaseFood = AnalyseBaseFood(root, group)
# BaseFood.compute_speed_food()
# BaseFood.compute_distance_to_food()
# BaseFood.compute_is_xy_next_to_food()
# BaseFood.compute_xy_next_to_food()
# BaseFood.compute_speed_xy_next_to_food()
# BaseFood.compute_speed_next_to_food()
# BaseFood.compute_orientation_next_to_food()
BaseFood.compute_orientation_to_food()

print(angle([0.5, 0.5])*180/pi)

# BaseFood.exp.plot_traj_on_movie(['xy_next_to_food', 'orientation_next_to_food'], 1, 2114)

# BaseFood.exp.load(['speed', 'speed_next_to_food', 'food_speed'])
#
# fig, ax = BaseFood.exp.food_speed.plotter.create_plot()
# BaseFood.exp.speed_next_to_food.plotter.hist1d(xscale='log', yscale='log', preplot=(fig, ax), normed=True)
# BaseFood.exp.food_speed.plotter.hist1d(xscale='log', yscale='log', preplot=(fig, ax), c='red', normed=True)
# BaseFood.exp.speed.plotter.hist1d(xscale='log', yscale='log', preplot=(fig, ax), c='blue', normed=True)
#
# plt.legend()
# plt.show()
