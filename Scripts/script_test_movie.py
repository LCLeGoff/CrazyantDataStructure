from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Movies.Movies import Movies
from Scripts.root import root
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

group = 'UO'
# mov.display()

id_exp = 3
for id_exp in range(4, 10):

    Exps = ExperimentGroupBuilder(root).build(group)
    session, trial = Exps.get_ref_id_exp(id_exp)
    print(session, trial)
    mov = Movies('/data/POSTDOC/CrazyAnt/UO/UO_S'+str(session).zfill(2)+'_T'+str(trial).zfill(2)+'.MP4')

    frame0 = min(Exps.timeseries_exp_frame_index[id_exp])

    for frame in frame0+np.arange(0, 1):
        frame_img = mov.get_frame(frame+1)
        plt.figure(figsize=(12, 12))
        plt.imshow(frame_img, cmap='gray')
        for id_ant in Exps.timeseries_exp_frame_ant_index[id_exp][frame]:

            x, y = Exps.convert_xy_to_movie_system(id_exp, id_ant, frame)
            orientation = Exps.convert_orientation_to_movie_system(id_exp, id_ant, frame)
            lg = 10
            plt.plot(x, y, '.')
            plt.plot(x+np.array([-1, 1])*lg*np.cos(orientation)/2., y+np.array([-1, 1])*lg*np.sin(orientation)/2.)

            # x, y = Exps.convert_xy_to_movie_system(id_exp, id_ant, pd.IndexSlice[frame0:frame])
            # plt.plot(x, y)

        plt.title(frame)
        plt.xlim((1000, 1200))
        plt.ylim((450, 650))
        plt.show()
