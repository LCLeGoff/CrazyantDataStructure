import itertools

import numpy as np
import pandas as pd
import pylab as pb
import scipy.cluster.hierarchy as sch
import random as rd

from rdp import rdp
from sklearn import svm
from tslearn.clustering import TimeSeriesKMeans

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from DataStructure.VariableNames import id_frame_name, id_exp_name
from Scripts.root import root
from Tools.MiscellaneousTools.Geometry import angle, get_line, get_line_tab

group = 'UO'

Exps = ExperimentGroupBuilder(root).build(group)


name_speed = 'mm1s_food_speed_leader_feature'
name_rotation = 'mm10_food_rotation_leader_feature'
name_orientation = 'mm1s_food_orientation_leader_feature'
name_attachment_angle = 'mm1s_attachment_angle_leader_feature'
name_confidence = 'food_confidence_leader_feature'

name_attachment = 'manual_leading_attachments'
Exps.load([name_speed, name_attachment_angle, name_attachment, name_rotation])

# Exps.load([name_speed, name_rotation, name_attachment, name_orientation, name_attachment_angle, name_confidence])
# # id_exp = 30
# # df_speed = Exps.get_df(name_speed).loc[id_exp, :]
# # df_rotation = Exps.get_df(name_rotation).loc[id_exp, :]
# # df_orientation = Exps.get_df(name_orientation).loc[id_exp, :]
# # df_attachment_angle = Exps.get_df(name_attachment_angle).loc[id_exp, :]
# # df_confidence = Exps.get_df(name_confidence).loc[id_exp, :]
#
#
# def linearize(tab, eps, min_angle, min_dt2):
#     mask = np.where(~np.isnan(tab[:, 1]))[0]
#     tab = tab[mask, :]
#     r = rdp(tab, epsilon=eps)
#
#     if len(r) > 2:
#         dr = r[1:, :] - r[:-1, :]
#         mask = list(np.where(dr[:, 0] > min_dt2)[0])
#         if 0 not in mask:
#             if 1 in mask:
#                 mask.remove(1)
#             mask = [0] + mask
#         mask.sort()
#         r = r[mask + [len(r) - 1], :]
#
#     if len(r) > 2:
#         dr = r[1:, :] - r[:-1, :]
#         a = angle(dr[:-1, :], dr[1:, :])
#         list_large_angle = np.where(np.abs(a) < min_angle)[0]
#         mask = list(range(len(r)))
#         [mask.remove(e + 1) for e in list_large_angle]
#
#         if 0 not in mask:
#             if 1 in mask:
#                 mask.remove(1)
#             mask = [0] + mask
#         mask.sort()
#         r = r[mask, :]
#
#     return r
#


def add_t(is_leader):
    ll = Exps.get_df(name_attachment).loc[id_exp, :, :]
    ll = ll[ll == is_leader].dropna()
    ll = np.array(ll.index.get_level_values('frame'))
    ll = set(np.concatenate([ll + dt for dt in range(-3, 4)]))
    ll = set(ll.intersection(set_frames))
    list_t[0] += [id_exp]*len(ll)
    list_t[1] += list(ll)
    return len(ll)


list_t = [[], []]
labels = []

for id_exp in set(Exps.get_index(name_attachment).get_level_values(id_exp_name)):
    set_frames = set(Exps.get_df(name_speed).loc[id_exp, :, :].index.get_level_values('frame'))
    lg = add_t(1)
    labels += [1]*lg
    lg = add_t(0)
    labels += [0]*lg

list_t = np.array(list_t).T


def cost_function(xs, arr):
    lg = len(arr)
    xs2 = [0]+list(xs)+[lg - 1]
    s = 0
    a = []
    b = []

    for k in range(1, arr.shape[1]):
        pts = arr[:, [0, k]]
        a2, b2 = get_line_tab(pts[xs2[:-1], :], pts[xs2[1:], :])
        for i in range(len(xs2)-1):
            s2 = np.sum((a2[i]*arr[xs2[i]:xs2[i+1]+1, 0]+b2[i]-arr[xs2[i]:xs2[i+1]+1, k])**2)
            s += s2
        a.append(a2)
        b.append(b2)
    return s, a, b


def j_function(xs, arr, a, b, di):
    j_fct = np.zeros(len(xs))
    for i in range(len(xs)):
        x0 = arr[xs[i]-di, 0]
        x2 = arr[xs[i]+di, 0]
        for k in range(0, arr.shape[1]-1):
            y0 = arr[xs[i]-di, k+1]
            y2 = arr[xs[i]+di, k+1]
            a2 = a[k]
            b2 = b[k]
            j_fct[i] += (a2[i]*x2+b2[i]-y2)**2-(a2[i]*x0+b2[i]-y0)**2
    return j_fct


def linearize2(arr, n_piece=5, di=20):
    arr = arr[~np.isnan(arr[:, 1]), :]
    lg = len(arr)
    combs = list(itertools.combinations(np.arange(1, int(lg / di) - 1)*di, n_piece-1))
    xs = list(combs[0])
    xs3 = [0]+list(xs)+[lg-1]

    s_min, a, b = cost_function(xs, arr)
    # j_fct = j_function(xs, arr, a, b, di)

    # fig, ax = pb.subplots(2)
    #
    # ax[0].plot(arr[:, 0], arr[:, 1])
    # g0, = ax[0].plot(arr[xs3, 0], arr[xs3, 1], marker='o')
    # g2, = ax[0].plot(arr[xs3, 0], arr[xs3, 1], marker='o')
    # pb.pause(0.25)
    # ax[1].plot(arr[:, 0], arr[:, 2])
    # g1, = ax[1].plot(arr[xs3, 0], arr[xs3, 2], marker='o')
    # fig.suptitle((0, round(s_min, 2)))
    # print((0, round(s_min, 2)))

    for i, xs2 in enumerate(combs):
        # dx = np.random.randint(-1, 2, n_piece-1)*di
        # xs2 = xs + dx
        # xs2 = np.maximum(xs2, di)
        # xs2 = np.minimum(xs2, lg-di)
        # xs2 = list(xs2)
        # xs2.sort()
        # xs2 = list(np.array(rd.sample(range(int(lg/di)), 4))*di)
        # xs2.sort()
        # g2.set_xdata(arr[[0] + list(xs2) + [lg - 1], 0])
        # g2.set_ydata(arr[[0] + list(xs2) + [lg - 1], 1])

        s, a, b = cost_function(xs2, arr)
        if s < s_min:
            s_min = s
            xs = xs2

            xs3 = [0]+list(xs)+[lg-1]
            # g0.set_xdata(arr[xs3, 0])
            # g0.set_ydata(arr[xs3, 1])
            # g1.set_xdata(arr[xs3, 0])
            # g1.set_ydata(arr[xs3, 2])

            # fig.suptitle((i, round(s_min, 2)))
            # pb.pause(0.25)
        #
        # print((i, round(s_min, 2)))

    return xs3


window = 20
n_piece = 5
di = 30
features = []
features_a_angle = []
features_angle = []
features_angle_012 = []
features_a_speed = []
features_speed = []
features_a_rotation = []
x_features = []
labels2 = []
list_t_features = []
for i, (id_exp, f) in enumerate(list_t):
    print(i, id_exp, f)
    arr = np.zeros((601, 4))
    df_temp = Exps.get_df(name_attachment_angle).loc[id_exp, :, f].transpose().abs()
    df_temp = df_temp.rolling(center=True, window=window).mean()
    df_temp = df_temp.reset_index().astype(float).values
    min_angle = np.nanmin(df_temp[:, 1])
    df_temp[:, 1] -= min_angle
    max_angle = np.nanmax(df_temp[:, 1])
    df_temp[:, 1] /= max_angle
    arr[:, :2] = df_temp

    df_temp = Exps.get_df(name_speed).loc[id_exp, :, f].transpose()
    df_temp = df_temp.rolling(center=True, window=window).mean()
    df_temp = df_temp.reset_index().astype(float).values
    min_speed = np.nanmin(df_temp[:, 1])
    df_temp[:, 1] -= min_speed
    max_speed = np.nanmax(df_temp[:, 1])
    df_temp[:, 1] /= max_speed
    arr[:, 2] = df_temp[:, 1]

    df_temp = Exps.get_df(name_rotation).loc[id_exp, :, f].transpose().abs()
    df_temp = df_temp.rolling(center=True, window=window).mean()
    df_temp = df_temp.reset_index().astype(float).values
    df_temp[:, 1] -= np.nanmin(df_temp[:, 1])
    df_temp[:, 1] /= np.nanmax(df_temp[:, 1])
    arr[:, 3] = df_temp[:, 1]

    arr = arr[~np.isnan(arr[:, 1]), :]
    arr = arr[~np.isnan(arr[:, 2]), :]
    if len(arr) > di*n_piece:
        xs = linearize2(arr[:, :3], n_piece=n_piece, di=di)

        features_angle.append(list(arr[xs, 1]*max_angle+min_angle))
        features_speed.append(list(arr[xs, 2]*max_speed+min_speed))

        x_features.append(list(arr[xs, 0]))

        pts = arr[:, :2]
        a, _ = get_line_tab(pts[xs[:-1], :], pts[xs[1:], :])
        feature = list(a)
        features_a_angle.append(list(a))

        ang = pts[xs, 1]
        ang2 = pts[xs, 1]
        ang2 *= max_angle
        ang2 += min_angle
        ang2 = (ang2 < 1.1).astype(int)
        ang = ang2[1:]+ang2[:-1]
        ang *= ang2[1:]
        feature += list(ang)
        features_angle_012.append(ang)

        pts = arr[:, [0, 2]]
        a, _ = get_line_tab(pts[xs[:-1], :], pts[xs[1:], :])
        feature += list(a)
        features_a_speed.append(list(a))

        pts = arr[:, [0, 3]]
        a, _ = get_line_tab(pts[xs[:-1], :], pts[xs[1:], :])
        feature += list(a)
        features_a_rotation.append(list(a))

        features.append(feature)
        labels2.append(labels[i])
        list_t_features.append((id_exp, f))

features = np.array(features)
features_a_angle = np.array(features_a_angle)
features_angle_012 = np.array(features_angle_012)
features_a_speed = np.array(features_a_speed)
features_a_rotation = np.array(features_a_rotation)

x_features = np.array(x_features)
features_angle = np.array(features_angle)
features_speed = np.array(features_speed)
labels2 = np.c_[labels2]
list_t_features = np.array(list_t_features)


# fig, ax0 = pb.subplots()
# fig, ax1 = pb.subplots()
# fig, ax2 = pb.subplots()
# fig, ax3 = pb.subplots()
# cat0 = []
# cat1 = []
# cat2 = []
# cat3 = []
#
# for i in range(len(labels2)):
#     j = np.where((features1[i, 1:-1] == 1))[0]
#     if len(j) == 0:
#         j = np.where((features1[i, 1:-1] == 2))[0]
#         if len(j) == 0:
#             cat0.append(i)
#             ax0.plot(features2[i, 0], features2[i, 1], '^', mec='k', c=str(labels2[i][0]))
#         else:
#             cat1.append(i)
#             j = j[0]+1
#             ax1.plot(features2[i, j]*(x_features[i, j+1]-x_features[i, j]), features2[i, j + 1]*(x_features[i, j+2]-x_features[i, j+1]), 's', mec='k', c=str(labels2[i][0]))
#     else:
#         j = np.where((features1[i, 1:-1] == 1)*(features1[i, 2:] == 2))[0]
#         if len(j) == 0:
#             cat2.append(i)
#             ax2.plot(features2[i, 0], features2[i, 1], 'P', mec='k', c=str(labels2[i][0]))
#         else:
#             cat3.append(i)
#             j = j[0]+1
#             ax3.plot(features2[i, j], features2[i, j+1], 'o', mec='k', c=str(labels2[i][0]))

def plot(i):
    if isinstance(i, tuple):
        id_exp, f = i
    else:
        id_exp, f = list_t_features[i]
        i = np.where(np.sum(np.array(list_t_features) == [id_exp, f], axis=1) == 2)[0][0]
        print(i)
    arr = np.zeros((601, 3))

    df_temp = Exps.get_df(name_attachment_angle).loc[id_exp, :, f].transpose().abs()
    df_temp = df_temp.rolling(center=True, window=window).mean()
    df_temp = df_temp.reset_index().astype(float).values
    arr[:, :2] = df_temp

    df_temp = Exps.get_df(name_speed).loc[id_exp, :, f].transpose()
    df_temp = df_temp.rolling(center=True, window=window).mean()
    df_temp = df_temp.reset_index().astype(float).values
    arr[:, 2] = df_temp[:, 1]

    arr = arr[~np.isnan(arr[:, 1]), :]
    arr = arr[~np.isnan(arr[:, 2]), :]

    df_temp = Exps.get_df(name_rotation).loc[id_exp, :, f].transpose().abs()
    df_temp = df_temp.rolling(center=True, window=window).mean()
    df_temp = df_temp.reset_index().astype(float).values

    if labels2[i] == 1:
        c = 'r'
    else:
        c = 'k'
    fig, ax = pb.subplots(3)
    fig.suptitle((i, (id_exp, f)))
    ax[0].plot(arr[:, 0], arr[:, 1], c=c)
    ax[1].plot(arr[:, 0], arr[:, 2], c=c)
    ax[2].plot(df_temp[:, 0], df_temp[:, 1], c=c)
    for x in x_features[i]:
        print(x)
        ax[0].axvline(x, ls='--', c='gray')
        ax[1].axvline(x, ls='--', c='gray')
        ax[2].axvline(x, ls='--', c='gray')
    ax[0].set_xlim(-3, 3)
    ax[0].set_ylim(0, 3.14)
    ax[0].grid()
    ax[1].set_xlim(-3, 3)
    ax[1].set_ylim(0, 10)
    ax[1].grid()
    ax[2].set_xlim(-3, 3)
    ax[2].set_ylim(0, 3.14)
    ax[2].grid()

    # pb.show()


fig, ax = pb.subplots(2, 3, figsize=(10, 10))
for k1 in range(ax.shape[0]):
    for k2 in range(ax.shape[1]):
        ax[k1, k2].axvline(0, ls='--', c='gray')
        ax[k1, k2].axhline(0, ls='--', c='gray')

for i in range(len(labels2)):
    if np.sum(features_angle_012[i, 1:-1]) == 2*(n_piece-2):
        j = np.argmax(features_a_angle[i, 1:-1])
        j += 1
        c = str(labels2[i][0])
        dx = x_features[i, 1:]-x_features[i, :-1]
        ax[0, 0].plot(features_a_speed[i, j], features_a_speed[i, j+1], 'o', mec='k', c=c)
        ax[0, 1].plot(features_a_speed[i, j]*dx[j], features_a_speed[i, j+1]*dx[j+1], 'o', mec='k', c=c)
        ax[0, 2].plot(features_speed[i, j], features_speed[i, j+1], 'o', mec='k', c=c)
        ax[1, 0].plot(features_speed[i, j+1], features_a_speed[i, j+1], 'o', mec='k', c=c)
        ax[1, 1].plot(features_speed[i, j+1], features_speed[i, j+2]-features_speed[i, j+1], 'o', mec='k', c=c)
        ax[1, 2].plot(features_a_speed[i, j+1], features_speed[i, j+2]-features_speed[i, j+1], 'o', mec='k', c=c)

pb.show()


# fig, ax = pb.subplots(3)
# ax[0].plot(arr[:, 0], arr[:, 1])
# ax[0].plot(arr[xs, 0], arr[xs, 1], marker='o')
# ax[1].plot(arr[:, 0], arr[:, 2])
# ax[1].plot(arr[xs, 0], arr[xs, 2], marker='o')
# ax[2].plot(arr[:, 0], arr[:, 3])
# ax[2].plot(arr[xs, 0], arr[xs, 3], marker='o')
# pb.show()


# window = 20
# min_dt = 0.3
#
#
# def add_slope(tab, j, k, tab_res):
#     id0 = np.where(tab[:, 0] < 0)[0]
#     if len(id0) > 0:
#         id0 = id0[-1]
#
#         a0, _ = get_line(tab[id0, :], tab[id0 + 1, :])
#         tab_res[j, 1 + k] = a0
#
#         if id0 == 0:
#             tab_res[j, 0 + k] = a0
#         else:
#             tab_res[j, 0 + k] = get_line(tab[id0 - 1, :], tab[id0, :])[0]
#
#         if id0 == len(tab) - 2:
#             tab_res[j, 2 + k] = a0
#         else:
#             tab_res[j, 2 + k] = get_line(tab[id0 + 1, :], tab[id0 + 2, :])[0]
#
#
# def add_data(tab, j, k, tab_res):
#     id0 = np.where(tab[:, 0] < 0)[0]
#     if len(id0) > 0:
#         id0 = id0[-1]
#
#         tab_res[j, 1 + k] = tab[id0, 1]
#
#         if id0 == 0:
#             tab_res[j, 0 + k] = tab[id0, 1]
#         else:
#             tab_res[j, 0 + k] = tab[id0-1, 1]
#
#         if id0 == len(tab) - 2:
#             tab_res[j, 2 + k] = tab[id0, 1]
#         else:
#             tab_res[j, 2 + k] = tab[id0+2, 1]
#
#
# m = len(labels)
# res = np.zeros((m, 3 * 4))
# to_remove = []
# for i, (id_exp, f) in enumerate(list_t):
#     print(i, id_exp, f)
#     df_temp = Exps.get_df(name_speed).loc[id_exp, :, f].transpose()
#     df_temp = df_temp.rolling(center=True, window=window).mean()
#     arr = df_temp.reset_index().astype(float).values
#     mask = np.where(~np.isnan(arr[:, 1]))[0]
#     if len(mask) == 0:
#         to_remove.append(i)
#     else:
#         arr = arr[mask, :]
#         arr = linearize(arr, 0.25, 0.5, min_dt)
#         add_slope(arr, i, 0, res)
#
#         df_temp = Exps.get_df(name_attachment_angle).loc[id_exp, :, f].transpose()
#         df_temp = df_temp.rolling(center=True, window=window).mean()
#         arr = df_temp.reset_index().astype(float).values
#         arr = linearize(arr, 0.25, 0.75, min_dt)
#         add_slope(arr, i, 3, res)
#         arr2 = arr.copy()
#         arr2[:, 1] = 0
#         arr2[:-1, 1] += arr[:-1, 1] > 1.1
#         arr2[:-1, 1] += arr[1:, 1] > 1.1
#         add_data(arr2, i, 6, res)
#
#         df_temp = Exps.get_df(name_rotation).loc[id_exp, :, f].transpose()
#         df_temp = df_temp.rolling(center=True, window=window).mean()
#         arr = df_temp.reset_index().astype(float).values
#         arr = linearize(arr, 0.1, 0.1, min_dt)
#         add_slope(arr, i, 9, res)
#
# labels2 = np.delete(labels, to_remove, 0)
# res = np.delete(res, to_remove, 0)

mask = np.where()

m = len(labels2)
sh = np.random.permutation(m)
features_svm = features[sh, :]
label_svm = labels2[sh]

m2 = int(m / 3 * 2)
features_training = features_svm[:m2, :]
features_val = features_svm[m2:, :]
labels_training = label_svm[:m2]
labels_val = label_svm[m2:]

best_f1 = 0

for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    for weight in [None, 'balanced']:
        for C in [0.01, 0.05, 0.1, 0.5, 1, 5, 10]:

            clf = svm.SVC(kernel=kernel, gamma='auto', C=C, class_weight=weight)
            clf.fit(features_training, labels_training)
            prediction = clf.predict(features_val)
            tp = sum(prediction*labels_val)
            fp = sum(prediction*(1-labels_val))
            fn = sum((1-prediction)*labels_val)
            tn = sum((1-prediction)*(1-labels_val))
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1 = 2*precision*recall/(precision+recall)
            accuracy = (tp+tn)/len(prediction)
            if best_f1 < f1:
                best_f1 = f1
                model = 'kernel=%s, weight=%s, C=%.2f' % (kernel, weight, C)
                mes = [precision, recall, accuracy, f1]
print(model)
print('  precision: %f,\n  recall: %f,\n  accuracy: %f,\n  f1: %f,\n ' % tuple(mes))


# name_food_speed = 'mm1s_food_speed'
# name_food_rotation = 'food_rotation'
# name_attachment = 'ant_attachment_intervals'
# name_outside_attachment = 'outside_ant_attachment_intervals'
# Exps.load([name_food_speed, name_food_rotation, name_attachment, name_outside_attachment])
#

# id_exp = 30
# df_speed = Exps.get_df(name_food_speed).loc[id_exp, :]
# df_food_rotation = Exps.get_df(name_food_rotation).loc[id_exp, :]
# df_food_rotation = df_food_rotation.reindex(df_speed.index)
#
# df_speed_diff = df_speed.copy()
# df_speed_diff.index -= 1
# df_speed_diff -= df_speed
# df_speed_diff = df_speed_diff.reindex(df_speed.index)
# df_rotation_diff = df_food_rotation.copy()
# df_rotation_diff.index -= 1
# df_rotation_diff -= df_food_rotation
# df_rotation_diff = df_rotation_diff.reindex(df_food_rotation.index)
#
# attachment_frames = Exps.get_df(name_attachment).loc[id_exp, :].index.get_level_values(id_frame_name)
# attachment_frames = np.sort(np.array(list(set(attachment_frames))))
# print(
#     len(attachment_frames),
#     len(Exps.get_df(name_outside_attachment).loc[id_exp, :].index.get_level_values(id_frame_name)))
# print(attachment_frames)
# print(list(Exps.get_df(name_outside_attachment).loc[id_exp, :].index.get_level_values(id_frame_name)))
#
# # dt = 200
# # for t in range(1549, 29549, 200):
# #     print(t-dt, t+dt)
# #     print(attachment_frames[(attachment_frames >= t-dt)*(attachment_frames <= t+dt)])
# #     print()
#
# dframes = attachment_frames[1:]-attachment_frames[:-1]
# # attachment_frames = attachment_frames[np.where(dframes > 100)[0]]
# # attachment_frames = attachment_frames[:-1]
#
#
# # list_dfs = [df_speed, df_angular_speed, df_speed_diff, df_angular_speed_diff]
# # res = np.zeros((len(attachment_frames), 800, len(list_dfs)))
# # for k, df in enumerate(list_dfs):
# #     for i, frame in enumerate(attachment_frames):
# #         ts = df.loc[frame:frame + 799].values.ravel()
# #         res[i, :len(ts), k] = ts
# # res[np.isnan(res)] = 0
#
# # from tslearn.clustering import TimeSeriesKMeans
# #
# # n_clusters = 5
# # km = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", max_iter=5, random_state=0).fit(res)
# # for i in range(n_clusters):
# #     fig, ax = pb.subplots(len(list_dfs))
# #     list_j = np.where(km.labels_ == i)[0]
# #     print(list_j)
# #     for j in list_j:
# #         for k in range(len(list_dfs)):
# #             ax[k].plot(res[j, :, k])
# #
# #
# # attachment_influencing = np.array(list('000010100010010001001000011000001011000010000'), dtype=int)
# # attachment_frame_influencing = attachment_frames[np.where(attachment_influencing == 1)[0]]
# #
# #
# # for j in range(n_clusters):
# #     frames = attachment_frames[np.where(km.labels_ == j)[0]]
# #     print(j, frames)
# #     print([frame in attachment_frame_influencing for frame in frames])
# #     print()
#
#

# def linearize(arr, eps, dy_eps, min_dt):
#     mask = np.where(~np.isnan(arr[:, 1]))[0]
#     arr2 = arr[mask, :]
#     r = rdp(arr2, epsilon=eps)
#
#     dr = r[1:, :] - r[:-1, :]
#     mask = list(np.where(dr[:, 0] > min_dt)[0])
#     if 0 not in mask:
#         if 1 in mask:
#             mask.remove(1)
#         mask = [0] + mask
#     mask.sort()
#     r = r[mask + [len(r) - 1], :]
#
#     dr = r[1:, :] - r[:-1, :]
#     a = dr[:, 1]
#     mask = np.where(np.abs(a) > dy_eps)[0]
#     mask = list(set(list(mask) + list(mask + 1)))
#
#     if 0 not in mask:
#         if 1 in mask:
#             mask.remove(1)
#         mask = [0] + mask
#     mask.sort()
#     r = r[mask + [len(r) - 1], :]
#
#     if len(r) > 2:
#         dr = r[1:, :] - r[:-1, :]
#         a = angle(dr[:-1, :], dr[1:, :])
#         mask = np.where(np.abs(a) > 0.1)[0]
#         mask = list(set(list(mask) + list(mask + 1)))
#         if 0 not in mask:
#             if 1 in mask:
#                 mask.remove(1)
#             mask = [0] + mask
#         mask.sort()
#         r = r[mask + [len(r) - 1], :]
#
#     return r
#
#
# name = 'mm1s_attachment_angle_leader_feature'
# Exps.load(name)
# arr = Exps.get_df(name).loc[id_exp, :].loc[pd.IndexSlice[:, 3784], :].transpose().reset_index().astype(float).values
# print(linearize(arr, 0.2, 1, 0.1))
