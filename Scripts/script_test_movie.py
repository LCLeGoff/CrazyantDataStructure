from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Movies.Movies import Movies
from Scripts.root import root
import matplotlib.pyplot as plt
import numpy as np

group = 'UO'
mov = Movies('/data/POSTDOC/CrazyAnt/UO/UO_S01_T01.MP4')
# mov.display()
id_exp = 1
frame0 = 2091
id_ant = 2

Exps = ExperimentGroupBuilder(root).build(group)
# Exps.load('major_axis_length')

for frame in frame0+np.append(-2, 3):
    frame_img = mov.get_frame(frame+1)
    plt.figure(figsize=(12, 12))
    plt.imshow(frame_img, cmap='gray')
    for id_ant in range(1, 16):

        x, y = Exps.convert_xy_to_movie_system(id_exp, id_ant, frame)
        orientation = -Exps.convert_orientation_to_movie_system(id_exp, id_ant, frame)

        # lg = float(Exps.major_axis_length.df.loc[id_exp, id_ant, frame])
        lg = 10

        plt.plot(x, y, '.')
        plt.plot(x+np.array([-1, 1])*lg*np.cos(orientation)/2., y+np.array([-1, 1])*lg*np.sin(orientation)/2.)
    plt.title(frame)
    plt.xlim((1000, 1200))
    plt.ylim((450, 650))
    plt.show()
