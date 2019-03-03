import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.preprocessing import MinMaxScaler

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Scripts.root import root

group = 'UO'

Exps = ExperimentGroupBuilder(root).build(group)
# Exps.load(
#     ['carrying_training_set', 'distance2food_next2food',
#      'distance2food_next2food_diff', 'speed_next2food', 'angle_body_food'])
#
# Exps.filter_with_time_occurrences(
#     name_to_filter='distance2food_next2food', filter_name='carrying_training_set',
#     result_name='training_set_distance', replace=True)
# Exps.filter_with_time_occurrences(
#     name_to_filter='distance2food_next2food_differential', filter_name='carrying_training_set',
#     result_name='training_set_distance_differential', replace=True)
# Exps.filter_with_time_occurrences(
#     name_to_filter='speed_next2food', filter_name='carrying_training_set',
#     result_name='training_set_speed', replace=True)
# Exps.filter_with_time_occurrences(
#     name_to_filter='angle_body_food', filter_name='carrying_training_set',
#     result_name='training_set_orientation', replace=True)
#
#
# distances = Exps.training_set_distance.get_array()
# distances_differential = np.abs(Exps.training_set_distance_differential.get_array())
# speeds = Exps.training_set_speed.get_array()
# orientations = np.abs(Exps.training_set_orientation.get_array())
# labels = np.array(Exps.carrying_training_set.get_values())
#
# mask = np.where(~(np.isnan(distances_differential))*~(np.isnan(distances))*~(np.isnan(speeds))*~(np.isnan(orientations)))[0]
# distances = distances[mask]
# distances_differential = distances_differential[mask]
# speeds = speeds[mask]
# orientations = orientations[mask]
# labels = labels[mask]
#
# # feature_names = ['distance', 'distance differential', 'orientation', 'speed']
# # features = np.array(list(zip(distances, distances_differential, orientations, speeds)))
# #
# # for i in range(4):
# #     for j in range(i+1, 4):
# #         plt.figure()
# #         plt.scatter(features[:, i], features[:, j], c=labels, cmap=plt.cm.Paired, edgecolors='k')
# #         plt.xlabel(feature_names[i])
# #         plt.ylabel(feature_names[j])
# #         if i == 3:
# #             plt.xscale('log')
# #             plt.xlim(min(features[features[:, i] != 0, i])*0.9, max(features[:, i])*1.1)
# #         elif j == 3:
# #             plt.yscale('log')
# #             plt.ylim(min(features[features[:, j] != 0, j])*0.9, max(features[:, j])*1.1)
# # plt.show()
#
#
# def predict(features, labels):
#     start = time.clock()
#     for i in range(5):
#         idx = np.arange(len(features))
#         np.random.shuffle(idx)
#         features2 = features[idx]
#         labels2 = labels[idx]
#         lg = int(len(features) / 2)
#
#         kernel = 'rbf'
#         clf = svm.SVC(kernel=kernel, gamma=2)
#         clf.fit(features2[:lg], labels2[:lg])
#         prediction = clf.predict(features2[lg:])
#         print(i, np.mean(prediction == labels2[lg:]))
#     print(time.clock() - start)
#
#
# features = np.array(list(zip(distances, orientations, speeds)))
# predict(features, labels)

# features = MinMaxScaler().fit_transform(np.array(list(zip(distances, orientations, speeds))))
# predict(features, labels)


# def index_random_training_set(features, nbr, lg):
#     res_trained = []
#     res2predict = []
#     for i in range(nbr):
#         idx = np.arange(len(features))
#         np.random.shuffle(idx)
#
#         res_trained.append(idx[:lg])
#         res2predict.append(idx[lg:])
#     return res_trained, res2predict
#
#
# features2 = MinMaxScaler().fit_transform(np.array(list(zip(distances, orientations))))
#
# c_range = [10**i for i in [-4, -3, -2, 0, 1, 2]]
# gamma_range = [10**i for i in [-4, -2, 0, 6, 12]]
# idx_trained, idx2predict = index_random_training_set(features2, 1, int(len(features2)/2))
# res = np.zeros((len(c_range), len(gamma_range)))
#
# for i, c in enumerate(c_range):
#     for j, gamma in enumerate(gamma_range):
#         print(i, j)
#         clf = svm.SVC(kernel='rbf', gamma=gamma, C=c)
#         clf.fit(features2[idx_trained[0]], labels[idx_trained[0]])
#         prediction = clf.predict(features2[idx2predict[0]])
#         res[i, j] = np.around(np.mean(prediction == labels[idx2predict[0]]), 3)
#
# plt.imshow(res, aspect='auto', extent=[c_range[0], c_range[-1], gamma_range[-1], gamma_range[0]])
# plt.colorbar()
# plt.show()

# load carrying label (training set), distance, speed and orientation
orientation_name = 'angle_body_food_next2food'
speed_name = 'speed_next2food'
distance_name = 'distance2food_next2food'
distance_diff_name = 'distance2food_next2food_diff'
training_set_name = 'carrying_training_set'

Exps.load([training_set_name, orientation_name, speed_name, distance_name, distance_diff_name])
# prepare arrays of labels and distance, speed and orientation features
Exps.filter_with_time_occurrences(
    name_to_filter=distance_name, filter_name='carrying_training_set',
    result_name='training_set_distance', replace=True)
Exps.filter_with_time_occurrences(
    name_to_filter=distance_diff_name, filter_name='carrying_training_set',
    result_name='training_set_distance_diff', replace=True)
Exps.filter_with_time_occurrences(
    name_to_filter=speed_name, filter_name='carrying_training_set',
    result_name='training_set_speed', replace=True)
Exps.filter_with_time_occurrences(
    name_to_filter=orientation_name, filter_name='carrying_training_set',
    result_name='training_set_orientation', replace=True)
Exps.filter_with_time_occurrences(
    name_to_filter=orientation_name, filter_name='carrying_training_set',
    result_name='training_set_id_exp', replace=True)
Exps.training_set_id_exp.df.training_set_id_exp = Exps.training_set_id_exp.df.index.get_level_values('id_exp')

features = Exps.training_set_speed.df.join(Exps.training_set_orientation.df, how='inner')
features = features.join(Exps.training_set_distance.df, how='inner')
features = features.join(Exps.training_set_distance_diff.df, how='inner')
features = features.join(Exps.training_set_id_exp.df, how='inner')
features.dropna(inplace=True)

labels = np.array(Exps.get_df(training_set_name).reindex(features.index))[:, 0]
features = np.array(features)

feature_names = np.array(['speed', 'orientation', 'distance', 'distance differential', 'id_exp'])

for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        plt.figure()
        plt.scatter(features[:, i], features[:, j], c=labels, cmap=plt.cm.Paired, edgecolors='k')
        plt.xlabel(feature_names[i])
        plt.ylabel(feature_names[j])
        if i == 3:
            plt.xscale('log')
            xmin = max(min(features[features[:, i] != 0, i])*0.9, 1e-2)
            plt.xlim(xmin, max(features[:, i])*1.1)
        if j ==3:
            plt.xscale('log')
            xmin = max(min(features[features[:, j] != 0, i])*0.9, 1e-2)
            plt.xlim(xmin, max(features[:, j])*1.1)
plt.show()