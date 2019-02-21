import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import time

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Scripts.root import root

group = 'UO'

Exps = ExperimentGroupBuilder(root).build(group)
Exps.load(
    ['carrying_training_set', 'distance_to_food_next_to_food',
     'distance_to_food_next_to_food_differential', 'speed_next_to_food', 'orientation_to_food'])

Exps.filter_with_time_occurrences(
    name_to_filter='distance_to_food_next_to_food', filter_name='carrying_training_set',
    result_name='training_set_distance', replace=True)
Exps.filter_with_time_occurrences(
    name_to_filter='distance_to_food_next_to_food_differential', filter_name='carrying_training_set',
    result_name='training_set_distance_differential', replace=True)
Exps.filter_with_time_occurrences(
    name_to_filter='speed_next_to_food', filter_name='carrying_training_set',
    result_name='training_set_speed', replace=True)
Exps.filter_with_time_occurrences(
    name_to_filter='orientation_to_food', filter_name='carrying_training_set',
    result_name='training_set_orientation', replace=True)


distances = Exps.training_set_distance.get_array()
distances_differential = np.abs(Exps.training_set_distance_differential.get_array())
speeds = Exps.training_set_speed.get_array()
orientations = np.abs(Exps.training_set_orientation.get_array())
labels = np.array(Exps.carrying_training_set.get_values())

mask = np.where(~(np.isnan(distances_differential))*~(np.isnan(distances))*~(np.isnan(speeds))*~(np.isnan(orientations)))[0]
distances = distances[mask]
distances_differential = distances_differential[mask]
speeds = speeds[mask]
orientations = orientations[mask]
labels = labels[mask]

feature_names = ['distance', 'distance differential', 'orientation', 'speed']
features = np.array(list(zip(distances, distances_differential, orientations, speeds)))

for i in range(4):
    for j in range(i+1, 4):
        plt.figure()
        plt.scatter(features[:, i], features[:, j], c=labels, cmap=plt.cm.Paired, edgecolors='k')
        plt.xlabel(feature_names[i])
        plt.ylabel(feature_names[j])
        if i == 3:
            plt.xscale('log')
            plt.xlim(min(features[features[:, i] != 0, i])*0.9, max(features[:, i])*1.1)
        elif j == 3:
            plt.yscale('log')
            plt.ylim(min(features[features[:, j] != 0, j])*0.9, max(features[:, j])*1.1)
plt.show()


# features = np.array(list(zip(distances, orientations, speeds)))
# start = time.clock()
# for i in range(5):
#     idx = np.arange(len(features))
#     np.random.shuffle(idx)
#     features = features[idx]
#     labels = labels[idx]
#     lg = int(len(features) / 2)
#
#     kernel = 'rbf'
#     clf = svm.SVC(kernel=kernel, gamma=2)
#     clf.fit(features[:lg], labels[:lg])
#     prediction = clf.predict(features[lg:])
#     print(i, np.mean(prediction == labels[lg:]))
# print(time.clock()-start)
