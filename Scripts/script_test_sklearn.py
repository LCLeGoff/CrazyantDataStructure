import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Scripts.root import root

group = 'UO'

Exps = ExperimentGroupBuilder(root).build(group)
Exps.load(['carrying_training_set', 'distance_to_food_next_to_food', 'speed_next_to_food', 'orientation_to_food'])
Exps.filter_with_time_occurrences(
    name_to_filter='distance_to_food_next_to_food', filter_name='carrying_training_set',
    result_name='training_set_distance')
Exps.filter_with_time_occurrences(
    name_to_filter='speed_next_to_food', filter_name='carrying_training_set',
    result_name='training_set_speed')
Exps.filter_with_time_occurrences(
    name_to_filter='orientation_to_food', filter_name='carrying_training_set',
    result_name='training_set_orientation')

distances = Exps.training_set_distance.get_array()
speeds = Exps.training_set_speed.get_array()
orientations = Exps.training_set_orientation.get_array()
labels = Exps.carrying_training_set.get_values()

plt.figure()
plt.scatter(
    distances, speeds, c=labels, zorder=10, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('distance')
plt.ylabel('speed')
plt.yscale('log')
plt.xlim((5, 15))
plt.ylim((1, 1e4))

plt.figure()
plt.scatter(
    distances, orientations, c=labels, zorder=10, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('distance')
plt.ylabel('orientation')
plt.xlim((5, 15))
plt.ylim((-np.pi, np.pi))

plt.figure()
plt.scatter(
    orientations, speeds, c=labels, zorder=10, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('orientation')
plt.ylabel('speed')
plt.yscale('log')
plt.xlim((-np.pi, np.pi))
plt.ylim((1, 1e4))

plt.show()

# # Our dataset and targets
# X = np.c_[
#     (.4, -.7), (-1.5, -1), (-1.4, -.9), (-1.3, -1.2), (-1.1, -.2), (-1.2, -.4),
#     (-.5, 1.2), (-1.5, 2.1), (1, 1),
#     (1.3, .8), (1.2, .5), (.2, -2), (.5, -2.4), (.2, -2.3), (0, -2.7), (1.3, 2.1)].T
# Y = [0] * 8 + [1] * 8
#
# # figure number
# fignum = 1
#
# # fit the model
# for kernel in ('linear', 'poly', 'rbf'):
#     clf = svm.SVC(kernel=kernel, gamma=2)
#     clf.fit(X, Y)
#
#     # plot the line, the points, and the nearest vectors to the plane
#     plt.figure(fignum, figsize=(4, 3))
#     plt.clf()
#
#     plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
#                 facecolors='none', zorder=10, edgecolors='k')
#     plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
#                 edgecolors='k')
#
#     plt.axis('tight')
#     x_min = -3
#     x_max = 3
#     y_min = -3
#     y_max = 3
#
#     XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
#     Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
#
#     # Put the result into a color plot
#     Z = Z.reshape(XX.shape)
#     plt.figure(fignum, figsize=(4, 3))
#     plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
#     plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
#                 levels=[-.5, 0, .5])
#
#     plt.xlim(x_min, x_max)
#     plt.ylim(y_min, y_max)
#
#     plt.xticks(())
#     plt.yticks(())
#     fignum = fignum + 1
# plt.show()
