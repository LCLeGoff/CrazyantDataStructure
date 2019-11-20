import numpy as np
import pandas as pd
import pylab as pb
import scipy.cluster.hierarchy as sch

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from DataStructure.VariableNames import id_frame_name
from Scripts.root import root
group = 'UO'

Exps = ExperimentGroupBuilder(root).build(group)
name_food_speed = 'mm1s_food_speed'
name_food_rotation = 'food_rotation'
name_attachment = 'ant_attachment_intervals'
name_outside_attachment = 'outside_ant_attachment_intervals'
Exps.load([name_food_speed, name_food_rotation, name_attachment, name_outside_attachment])

id_ant = 3
df_speed = Exps.get_df(name_food_speed).loc[id_ant, :]
df_food_rotation = Exps.get_df(name_food_rotation).loc[id_ant, :]
df_food_rotation = df_food_rotation.reindex(df_speed.index)

df_speed_diff = df_speed.copy()
df_speed_diff.index -= 1
df_speed_diff -= df_speed
df_speed_diff = df_speed_diff.reindex(df_speed.index)
df_rotation_diff = df_food_rotation.copy()
df_rotation_diff.index -= 1
df_rotation_diff -= df_food_rotation
df_rotation_diff = df_rotation_diff.reindex(df_food_rotation.index)

attachment_frames = Exps.get_df(name_attachment).loc[id_ant, :].index.get_level_values(id_frame_name)
attachment_frames = np.sort(np.array(list(set(attachment_frames))))
print(
    len(attachment_frames),
    len(Exps.get_df(name_outside_attachment).loc[id_ant, :].index.get_level_values(id_frame_name)))
print(attachment_frames)
print(list(Exps.get_df(name_outside_attachment).loc[id_ant, :].index.get_level_values(id_frame_name)))

# dt = 200
# for t in range(1549, 29549, 200):
#     print(t-dt, t+dt)
#     print(attachment_frames[(attachment_frames >= t-dt)*(attachment_frames <= t+dt)])
#     print()

dframes = attachment_frames[1:]-attachment_frames[:-1]
# attachment_frames = attachment_frames[np.where(dframes > 100)[0]]
# attachment_frames = attachment_frames[:-1]


# list_dfs = [df_speed, df_angular_speed, df_speed_diff, df_angular_speed_diff]
# res = np.zeros((len(attachment_frames), 800, len(list_dfs)))
# for k, df in enumerate(list_dfs):
#     for i, frame in enumerate(attachment_frames):
#         ts = df.loc[frame:frame + 799].values.ravel()
#         res[i, :len(ts), k] = ts
# res[np.isnan(res)] = 0

# from tslearn.clustering import TimeSeriesKMeans
#
# n_clusters = 5
# km = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", max_iter=5, random_state=0).fit(res)
# for i in range(n_clusters):
#     fig, ax = pb.subplots(len(list_dfs))
#     list_j = np.where(km.labels_ == i)[0]
#     print(list_j)
#     for j in list_j:
#         for k in range(len(list_dfs)):
#             ax[k].plot(res[j, :, k])
#
#
# attachment_influencing = np.array(list('000010100010010001001000011000001011000010000'), dtype=int)
# attachment_frame_influencing = attachment_frames[np.where(attachment_influencing == 1)[0]]
#
#
# for j in range(n_clusters):
#     frames = attachment_frames[np.where(km.labels_ == j)[0]]
#     print(j, frames)
#     print([frame in attachment_frame_influencing for frame in frames])
#     print()
