import itertools

import numpy as np
import pandas as pd
import pylab as pb
import scipy.cluster.hierarchy as sch
import random as rd

from rdp import rdp
from scipy.signal import argrelextrema
from sklearn import svm
from tslearn.clustering import TimeSeriesKMeans

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from DataStructure.VariableNames import id_frame_name, id_exp_name, id_ant_name
from Scripts.root import root
import Tools.MiscellaneousTools.Geometry as Geo

group = 'UO'

Exps = ExperimentGroupBuilder(root).build(group)

name_speed = 'speed'
name_body_orientation = 'mm10_orientation'
name_orientation = 'speed_phi'
name_radius = 'food_radius'
Exps.load([name_speed, name_body_orientation, name_orientation, name_radius])
name_xy = 'xy'
Exps.load_as_2d('mm10_x', 'mm10_y', name_xy, 'x', 'y', replace=True)
# name_food_xy = 'food_xy'
# Exps.load_as_2d('mm10_food_x', 'mm10_food_y', name_food_xy, 'x', 'y', replace=True)


def get_sequence(tab, dt=5):
    tab2 = tab.ravel()
    res = list(argrelextrema(tab2, np.less)[0])
    res += list(argrelextrema(tab2, np.greater)[0])
    res.sort()
    res = np.array(res)

    dx = res[1:] - res[:-1]
    mask = np.where(dx < dt)[0]
    while len(mask) != 0:
        ii = mask[0]
        if len(mask) > 1 and mask[1] == ii + 1:
            new_x = int(np.mean(res[[ii, ii + 2]]))
            res = np.delete(res, [ii, ii+1, ii+2])
            res = np.append(res, new_x)
            res = np.sort(res)
        else:
            res = np.delete(res, [ii, ii + 1])
        dx = res[1:] - res[:-1]
        mask = np.where(dx < dt)[0]
    res = list(set([0] + list(res) + [len(tab2) - 1]))
    res.sort()
    return res


id_exp = 30

da_max = 0.5
dv_min = 2
v_max = 5
djj = 5
dtheta_max = 90*np.pi/180
min_dist = 2

id_ants = np.array(Exps.get_index(name_speed).get_level_values(id_ant_name))
y, x = np.histogram(id_ants, range(1, max(id_ants)+1))
id_ants = list(x[:-1][y > 500])

res = []
for id_ant in id_ants:
    print(id_ant)
    df_speed0 = Exps.get_df(name_speed).loc[id_exp, id_ant, :].copy()/10.
    df_orient0 = Exps.get_df(name_orientation).loc[id_exp, id_ant, :].copy()/100.

    speed_tab = df_speed0.loc[id_exp, id_ant, :].dropna().reset_index().values
    orient_df = df_orient0.loc[id_exp, id_ant, :].dropna()

    orient_frames = set(orient_df.index)

    xs = np.array(get_sequence(speed_tab[:, 1], dt=5))
    if len(xs) > 3:
        a_speed, _ = Geo.get_line_tab(speed_tab[xs[:-1], :], speed_tab[xs[1:], :])

        da = np.abs(a_speed[1:]+a_speed[:-1])
        v = speed_tab[xs, :]
        dv = v[1:, 1]-v[:-1, 1]
        mask = list(np.where((a_speed[:-1] < 0)*(dv[:-1] < -dv_min)*(dv[1:] > dv_min)*(v[1:-1, 1] < v_max))[0]+1)

        for ii in mask:
            f0, f1, f2 = v[ii-1:ii+2, 0].astype(int)
            print(f1, f2-f0, round(v[ii, 1], 2), round(dv[ii-1], 2), round(dv[ii], 2), end=" ")
            if f2-f0 < 32:
                if {f0, f1, f2}.issubset(orient_frames):
                    theta0, theta1, theta2 = orient_df.loc[[f0, f1, f2]].values.ravel()
                    dtheta0 = Geo.angle_distance(theta0, theta1)
                    dtheta1 = Geo.angle_distance(theta1, theta2)
                    # dtheta = np.abs(dtheta0)+np.abs(dtheta1)
                    dtheta = np.abs(Geo.angle_distance(theta0, theta2))
                    print((round(dtheta0, 2), round(dtheta1, 2), round(dtheta, 2)), end=" ")

                    if dtheta < dtheta_max or (dtheta > np.pi and dtheta0*dtheta1 < 0):
                        xys = Exps.get_df(name_xy).loc[id_exp, :, f1]
                        xy = Exps.get_df(name_xy).loc[id_exp, id_ant, f1]
                        dist = Geo.squared_distance_df(xys, xy)
                        dist = round(dist.drop((id_exp, id_ant, f1)).min()/10., 1)
                        print(dist, end=" ")
                        if dist > min_dist:
                            res.append((id_exp, id_ant, f1))
            print()
    print()

df = pd.DataFrame(res, columns=[id_exp_name, id_ant_name, id_frame_name])
df.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)
res = np.array(res)
df['df'] = res[:, -1]

id_ants = list(set(res[:, 1]))
for id_ant in id_ants:
    df2 = df.loc[id_exp, id_ant, :].copy()
    if len(df2) < 5:
        df.loc[id_exp, id_ant, :] = np.nan
        df = df.dropna()


# def do4each_grooup(df: pd.DataFrame):
#     df.iloc[1:] = df.iloc[1:]-df.iloc[:-1]
#     df.iloc[0] =


print(len(res))
[print(e) for e in res]
#
# df_all = pd.read_csv(Exps.root + 'manual_markings.csv', index_col=[id_exp_name, id_ant_name, id_frame_name])
# df_all['algo2'] = np.nan
# c = []
# c2 = []
# for frame, v, df, dtheta, dist, algo in al:
#     df_all.loc[(id_exp, id_ant, frame), 'algo2'] = algo
#     c.append(df_all.loc[(id_exp, id_ant, frame), 'manual'])
#     c2.append(algo)
#
# tp = np.nansum(df_all.algo * df_all.manual)
# fp = np.nansum((1 - df_all.algo) * df_all.manual)
# tn = np.nansum((1 - df_all.algo) * (1 - df_all.manual))
# fn = np.nansum(df_all.algo * (1 - df_all.manual))
# print('TP:%i, FP:%i, TN:%i, FN:%i' % (tp, fp, tn, fn))
# accuracy = (tp+tn)/(tp+fp+tn+fn)
# precision = tp/(tp+fp)
# recall = tp/(tp+tn)
# f1 = 2*precision*recall/(precision+recall)
# print('accuracy:%.2f, precision:%.2f, recall:%.2f, f1:%.2f' % (accuracy, precision, recall, f1))
#
# tp = np.nansum(df_all.algo2 * df_all.manual)
# fp = np.nansum((1 - df_all.algo2) * df_all.manual)
# tn = np.nansum((1 - df_all.algo2) * (1 - df_all.manual))
# fn = np.nansum(df_all.algo2 * (1 - df_all.manual))
# print('TP:%i, FP:%i, TN:%i, FN:%i' % (tp, fp, tn, fn))
# accuracy = (tp+tn)/(tp+fp+tn+fn)
# precision = tp/(tp+fp)
# recall = tp/(tp+tn)
# f1 = 2*precision*recall/(precision+recall)
# print('accuracy:%.2f, precision:%.2f, recall:%.2f, f1:%.2f' % (accuracy, precision, recall, f1))
#
#
# labels = ['v', 'df', 'dtheta']
# c = np.array(c)*2
# c = np.where(np.isnan(c), 1, c)
# c2 = np.array(c2)*2
# c2 = np.where(np.isnan(c2), 1, c2)
# for i in range(len(labels)):
#     for j in range(i+1, len(labels)):
#         pb.subplots()
#         pb.subplot(121)
#         pb.scatter(al[:, i+1], al[:, j+1], c=c)
#         pb.title((labels[i], labels[j]))
#         pb.colorbar()
#         pb.subplot(122)
#         pb.scatter(al[:, i+1], al[:, j+1], c=c2)
#         pb.title((labels[i], labels[j]))
#         pb.colorbar()

# name_speed = 'mm1s_food_speed_leader_feature'
# name_rotation = 'mm10_food_rotation_leader_feature'
# name_orientation = 'mm1s_food_orientation_leader_feature'
# name_attachment_angle = 'mm1s_attachment_angle_leader_feature'
# name_confidence = 'food_confidence_leader_feature'
#
# name_manual_attachment = 'manual_leading_attachments'
# name_attachment = 'attachment_intervals'
# Exps.load([name_speed, name_attachment_angle, name_manual_attachment, name_rotation, name_attachment])
#
#
# def cost_function(xs3, tab):
#     lg = len(tab)
#     xs2 = [0] + list(xs3) + [lg - 1]
#     s = 0
#
#     for k in range(1, tab.shape[1]):
#         pts2 = tab[:, [0, k]]
#         a2, b2 = get_line_tab(pts2[xs2[:-1], :], pts2[xs2[1:], :])
#         for ii in range(len(xs2)-1):
#             s2 = np.sum((a2[ii] * tab[xs2[ii]:xs2[ii + 1] + 1, 0] + b2[ii] - tab[xs2[ii]:xs2[ii + 1] + 1, k]) ** 2)
#             s += s2
#     return s
#
#
# def linearize2(tab, n=5, d=20):
#     tab = tab[~np.isnan(tab[:, 1]), :]
#     lg = len(tab)
#     combs = list(itertools.combinations(np.arange(1, int(lg / d) - 1) * d, n - 1))
#     xs2 = list(combs[0])
#     xs3 = [0]+list(xs2)+[lg-1]
#
#     s_min = cost_function(xs2, tab)
#
#     for xs2 in combs:
#
#         s = cost_function(xs2, tab)
#         if s < s_min:
#             s_min = s
#             xs2 = xs2
#
#             xs3 = [0]+list(xs2)+[lg-1]
#
#     return xs3
#
#
# influence_angle = 1.1
#
#
# def plot(id_exp2, frame, x_features=None):
#     tab = np.zeros((601, 4))
#
#     df = Exps.get_df(name_attachment_angle).loc[id_exp2, :, frame].transpose().abs()
#     df = df.rolling(center=True, window=window).mean()
#     df = df.reset_index().astype(float).values
#     tab[:, :2] = df
#
#     df = Exps.get_df(name_speed).loc[id_exp2, :, frame].transpose()
#     df = df.rolling(center=True, window=window).mean()
#     df = df.reset_index().astype(float).values
#     tab[:, 2] = df[:, 1]
#
#     df = Exps.get_df(name_rotation).loc[id_exp2, :, frame].transpose().abs()
#     df = df.rolling(center=True, window=window).mean()
#     df = df.reset_index().astype(float).values
#     tab[:, 3] = df[:, 1]
#
#     tab = tab[~np.isnan(tab[:, 1]), :]
#     tab = tab[~np.isnan(tab[:, 2]), :]
#
#     fig, ax = pb.subplots(3)
#     c = 'k'
#     fig.suptitle((id_exp2, frame))
#     ax[0].plot(tab[:, 0], tab[:, 1], c=c)
#     ax[1].plot(tab[:, 0], tab[:, 2], c=c)
#     ax[2].plot(tab[:, 0], tab[:, 3], c=c)
#
#     if x_features is not None:
#         ax[0].plot(tab[x_features, 0], tab[x_features, 1], marker='o', c='gray')
#         ax[1].plot(tab[x_features, 0], tab[x_features, 2], marker='o', c='gray')
#         ax[2].plot(tab[x_features, 0], tab[x_features, 3], marker='o', c='gray')
#
#     ax[0].set_xlim(-3, 3)
#     ax[0].set_ylim(0, 3.14)
#     ax[0].axhline(influence_angle, ls='--', c='gray')
#     # ax[0].grid()
#     ax[1].set_xlim(-3, 3)
#     # ax[1].set_ylim(0, 10)
#     # ax[1].grid()
#     ax[2].set_xlim(-3, 3)
#     ax[2].set_ylim(0, 3.14)
#     # ax[2].grid()
#
#
# window = 20
# n_piece = 5
# di = 30
# id_exp = 30
# df_attachment = Exps.get_df(name_attachment).loc[id_exp, :, :]
# frames = list(df_attachment.index.get_level_values(id_frame_name))
# frames.sort()
# leader = []
# xs_list = {}
# for f in frames:
#     arr = np.zeros((601, 4))
#     df_temp = Exps.get_df(name_attachment_angle).loc[id_exp, :, f].transpose().abs()
#     df_temp = df_temp.rolling(center=True, window=window).mean()
#     df_temp = df_temp.reset_index().astype(float).values
#     min_angle = np.nanmin(df_temp[:, 1])
#     df_temp[:, 1] -= min_angle
#     max_angle = np.nanmax(df_temp[:, 1])
#     df_temp[:, 1] /= max_angle
#     arr[:, :2] = df_temp
#
#     df_temp = Exps.get_df(name_speed).loc[id_exp, :, f].transpose()
#     df_temp = df_temp.rolling(center=True, window=window).mean()
#     df_temp = df_temp.reset_index().astype(float).values
#     min_speed = np.nanmin(df_temp[:, 1])
#     df_temp[:, 1] -= min_speed
#     max_speed = np.nanmax(df_temp[:, 1])
#     df_temp[:, 1] /= max_speed
#     arr[:, 2] = df_temp[:, 1]
#
#     df_temp = Exps.get_df(name_rotation).loc[id_exp, :, f].transpose().abs()
#     df_temp = df_temp.rolling(center=True, window=window).mean()
#     df_temp = df_temp.reset_index().astype(float).values
#     min_rotation = np.nanmin(df_temp[:, 1])
#     df_temp[:, 1] -= min_rotation
#     max_rotation = np.nanmax(df_temp[:, 1])
#     df_temp[:, 1] /= max_rotation
#     arr[:, 3] = df_temp[:, 1]
#
#     mask = np.where((arr[:, 1] * max_angle + min_angle < influence_angle) * (arr[:, 2] * max_speed + min_speed > 2))[0]
#     if len(mask) != 0:
#         arr = arr[~np.isnan(arr[:, 1]), :]
#         arr = arr[~np.isnan(arr[:, 2]), :]
#         if len(arr) != 0 and arr[0, 0] < -1 and  arr[-1, 0] > 1.5:
#             xs = linearize2(arr[:, :3], n=n_piece, d=di)
#             xs_list[f] = xs
#
#             pts = arr[:, :2]
#             ang = pts[xs, 1]
#             ang2 = pts[xs, 1]
#             ang2 *= max_angle
#             ang2 += min_angle
#             ang2 = (ang2 < influence_angle).astype(int)
#             ang = ang2[1:]+ang2[:-1]
#             ang *= ang2[1:]
#
#             speed = arr[:, [0, 2]]
#             rotation = arr[:, [0, 3]]
#             speed[:, 1] *= max_speed
#             speed[:, 1] += min_speed
#             rotation[:, 1] *= max_rotation
#             rotation[:, 1] += min_rotation
#
#             mask = np.where((ang[1:-1] == 1)*(ang[2:] == 2))[0]
#             if len(mask) == 0:
#                 # s = np.sum(ang[1:-1])
#                 # if s >= 4:
#                 #     a_speed, _ = get_line_tab(speed[xs[:-1], :], speed[xs[1:], :])
#                 #     mask = np.where(a_speed > 0.7)[0]
#                 #     if len(mask) == 0:
#                 #         leader.append((f, 0))
#                 #     else:
#                 #         leader.append((f, 1))
#                 # else:
#                 leader.append((f, 0))
#             else:
#                 j = mask[0] + 1
#                 if 2 > arr[xs[j+1], 0] > -1:
#                     a_speed, _ = get_line_tab(speed[xs[:-1], :], speed[xs[1:], :])
#                     a_rotation, _ = get_line_tab(rotation[xs[:-1], :], rotation[xs[1:], :])
#                     if (a_speed[j] > 0.1 or a_speed[j+1] > 0.1 or a_rotation[j+1] > 0.1) and speed[xs[j+2], 1] > 2:
#                         leader.append((f, 1))
#                     else:
#                         leader.append((f, 0))
#                 else:
#                     leader.append((f, 0))
#
#         else:
#             leader.append((f, 0))
#     else:
#         leader.append((f, 0))
#     print(leader[-1])
# leader = np.array(leader)
#
# window = 20
# n_piece = 5
# di = 30
# l_below_1 = []
# list_xs = []
# speeds = []
# rotations = []
# list_a = []
# for id_exp in [12, 20, 30]:
#     frames = list(Exps.get_df(name_attachment).loc[id_exp, :, :].index.get_level_values(id_frame_name))
#     frames.sort()
#     for f in frames:
#         print(id_exp, f)
#         arr = np.zeros((601, 4))
#         df_temp = Exps.get_df(name_attachment_angle).loc[id_exp, :, f].transpose().abs()
#         df_temp = df_temp.rolling(center=True, window=window).mean()
#         df_temp = df_temp.reset_index().astype(float).values
#         min_angle = np.nanmin(df_temp[:, 1])
#         df_temp[:, 1] -= min_angle
#         max_angle = np.nanmax(df_temp[:, 1])
#         df_temp[:, 1] /= max_angle
#         arr[:, :2] = df_temp
#
#         df_temp = Exps.get_df(name_speed).loc[id_exp, :, f].transpose()
#         df_temp = df_temp.rolling(center=True, window=window).mean()
#         df_temp = df_temp.reset_index().astype(float).values
#         min_speed = np.nanmin(df_temp[:, 1])
#         df_temp[:, 1] -= min_speed
#         max_speed = np.nanmax(df_temp[:, 1])
#         df_temp[:, 1] /= max_speed
#         arr[:, 2] = df_temp[:, 1]
#
#         df_temp = Exps.get_df(name_rotation).loc[id_exp, :, f].transpose().abs()
#         df_temp = df_temp.rolling(center=True, window=window).mean()
#         df_temp = df_temp.reset_index().astype(float).values
#         min_rotation = np.nanmin(df_temp[:, 1])
#         df_temp[:, 1] -= min_rotation
#         max_rotation = np.nanmax(df_temp[:, 1])
#         df_temp[:, 1] /= max_rotation
#         arr[:, 3] = df_temp[:, 1]
#
#         mask = np.where((arr[:, 1] * max_angle + min_angle < influence_angle) * (arr[:, 2] * max_speed + min_speed > 2))[0]
#         if len(mask) != 0:
#             arr = arr[~np.isnan(arr[:, 1]), :]
#             arr = arr[~np.isnan(arr[:, 2]), :]
#             if len(arr) != 0 and arr[0, 0] < -1 and  arr[-1, 0] > 1.5:
#                 xs = linearize2(arr[:, :3], n=n_piece, d=di)
#
#                 pts = arr[:, :2]
#                 ang = pts[xs, 1]
#                 ang2 = pts[xs, 1]
#                 ang2 *= max_angle
#                 ang2 += min_angle
#                 ang2 = (ang2 < influence_angle).astype(int)
#                 ang = ang2[1:]+ang2[:-1]
#                 ang *= ang2[1:]
#
#                 speed = arr[:, [0, 2]]
#                 speed[:, 1] *= max_speed
#                 speed[:, 1] += min_speed
#
#                 rotation = arr[:, 3]
#                 rotation *= max_rotation
#                 rotation += min_rotation
#
#                 mask = np.where((ang[1:-1] == 1)*(ang[2:] == 2))[0]
#                 if len(mask) == 0:
#                     s = np.sum(ang[1:-1])
#                     if s >= 4:
#                         a_speed, _ = get_line_tab(speed[xs[:-1], :], speed[xs[1:], :])
#                         l_below_1.append((id_exp, f))
#                         list_a.append(a_speed)
#                         speeds.append(list(speed[xs, 1]))
#                         rotations.append(list(rotation[xs]))
#                         list_xs.append(speed[xs, 0])
#
# # l = [(12, 8869), (20, 16445), (20, 19580), (20, 23087), (45, 5604), (45, 6721), (47, 2933), (49, 9103)]
# l2 = [(20, 2203), (20, 12133), (20, 16445), (20, 20652), (20, 23087), (20, 23208), (12, 4180), (12, 6726),
#       (30, 3248), (30, 3374), (30, 5432), (30, 8000), (30, 11802), (30, 13038)]
# l3 = [(20, 4174), (20, 4857), (20, 6013), (12, 4180), (30, 3788)]
# l4 = [(20, 11576), (20, 17601), (20, 19581), (20, 19829), (20, 22260), (20, 23208), (20, 26116),
#       (12, 2272), (12, 9677), (12, 11937), (12, 12909), (30, 13238), (30, 17834), (30, 18653), (30, 21465)]
# l_below_1 = np.array(l_below_1)
# speeds = np.array(speeds)
# rotations = np.array(rotations)
# list_xs = np.array(list_xs)
# list_a = np.array(list_a)
# c = []
# for id_exp, frame in l_below_1:
#     c.append((int((id_exp, frame) in l3)*0.5+int((id_exp, frame) in l4)))
#
# idx = np.argmax(list_a[:, 1:], axis=1)+1
# d_speed_max = [speeds[i, idx[i]+1]-speeds[i, idx[i]] for i in range(len(speeds))]
# speed_max = [speeds[i, idx[i]] for i in range(len(speeds))]
# d_rotation_max = [rotations[i, idx[i]+1]-rotations[i, idx[i]] for i in range(len(speeds))]
# rotation_max = [rotations[i, idx[i]] for i in range(len(speeds))]
# xs_max = [list_xs[i, idx[i]] for i in range(len(speeds))]
# a_max = np.max(list_a, axis=1)
#
#
# tabs = [speed_max, d_speed_max, rotation_max, d_rotation_max, xs_max, a_max]
# labels = ['v', 'dv', 'rot', 'drot', 'x', 'a']
# for i in range(len(tabs)):
#     for j in range(i+1, len(tabs)):
#         pb.subplots()
#         pb.scatter(tabs[i], tabs[j], c=c)
#         pb.title(labels[i]+', '+labels[j])
#
# print()
#
