import numpy as np
import pandas as pd

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.MiscellaneousTools.Geometry import angle, distance

from Tools.Plotter.Plotter import Plotter


class AnalyseTrajectory(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'Trajectory'

    def compute_mm10_bodyLength(self):
        name = 'bodyLength'
        self.exp.load(name)
        time_window = 10

        result_name = self.exp.rolling_mean(
            name_to_average=name, window=time_window, category=self.category, is_angle=False)
        self.exp.write(result_name)

    def compute_r_phi(self):
        print('r, phi')
        self.exp.load(['x', 'y'])

        self.__merge_xy_in2d()
        self.__copy_xy_to_r_phi()

        r = np.around(distance([0, 0], self.exp.xy.get_array()), 3)
        phi = np.around(angle([1, 0], self.exp.xy.get_array()), 3)
        self.exp.r.replace_values(r)
        self.exp.phi.replace_values(phi)

        self.exp.write(['r', 'phi'])

    def __copy_xy_to_r_phi(self):
        self.exp.add_copy1d(
            name_to_copy='x', copy_name='r',
            category=self.category,
            label='r',
            description='radial coordinate (in the food system)'
        )
        self.exp.add_copy1d(
            name_to_copy='x', copy_name='phi',
            category=self.category,
            label='phi',
            description='angular coordinate (in the food system)'
        )

    def __merge_xy_in2d(self):
        self.exp.add_2d_from_1ds(
            name1='x', name2='y',
            result_name='xy', xname='x', yname='y',
            category=self.category, label='coordinates', xlabel='x', ylabel='y',
            description='coordinates of ant positions'
        )

    def compute_mm10_traj(self):
        name_x = 'x'
        name_y = 'y'
        self.exp.load([name_x, name_y])
        time_window = 10

        category = 'TrajMM'
        result_name = self.exp.rolling_mean(
            name_to_average=name_x, window=time_window, category=category, is_angle=False)
        self.exp.write(result_name)

        result_name = self.exp.rolling_mean(
            name_to_average=name_y, window=time_window, category=category, is_angle=False)
        self.exp.write(result_name)

    def compute_speed(self, redo=False, redo_hist=False):
        name = 'speed'
        hist_name = 'speed_hist'
        bins = np.arange(0, 8e2, 1)
        hist_label = 'Distribution of the speed (mm/s)'
        hist_description = 'Distribution of the instantaneous speed of the ants (mm/s)'

        if redo:
            name_x = 'mm10_x'
            name_y = 'mm10_y'
            self.exp.load_as_2d(name_x, name_y, result_name='xy', xname='x', yname='y', replace=True)
            self.exp.load('fps')

            self.exp.add_copy1d(
                name_to_copy=name_x, copy_name=name, category=self.category, label='Speed',
                description='Instantaneous speed of the ants'
            )
            self.exp.add_copy1d(
                name_to_copy=name_x, copy_name=name+'_phi', category=self.category, label='Speed',
                description='Angular coordinate of the instantaneous speed of the ants'
            )
            self.exp.add_copy1d(
                name_to_copy=name_x, copy_name=name + '_x', category=self.category, label='X speed',
                description='X coordinate of the instantaneous speed of the ants'
            )
            self.exp.add_copy1d(
                name_to_copy=name_x, copy_name=name + '_y', category=self.category, label='Y speed',
                description='Y coordinate of the instantaneous speed of the ants'
            )

            def compute_speed4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                id_ant = df.index.get_level_values(id_ant_name)[0]
                frames = df.index.get_level_values(id_frame_name)
                print(id_exp, id_ant)

                dx = np.array(df.x)
                dx1 = dx[1].copy()
                dx2 = dx[-2].copy()
                dx[1:-1] = (dx[2:]-dx[:-2])/2.
                dx[0] = dx1-dx[0]
                dx[-1] = dx[-1]-dx2

                dy = np.array(df.y)
                dy1 = dy[1].copy()
                dy2 = dy[-2].copy()
                dy[1:-1] = (dy[2:]-dy[:-2])/2.
                dy[0] = dy1-dy[0]
                dy[-1] = dy[-1]-dy2

                dt = np.array(frames, dtype=float)
                dt.sort()
                dt[1:-1] = dt[2:]-dt[:-2]
                dt[0] = 1
                dt[-1] = 1
                dx[dt > 2] = np.nan
                dy[dt > 2] = np.nan

                fps = self.exp.get_value('fps', id_exp)
                self.exp.speed_x.df.loc[id_exp, id_ant, :] = np.c_[np.around(dx * fps, 3)]
                self.exp.speed_y.df.loc[id_exp, id_ant, :] = np.c_[np.around(dy * fps, 3)]
                self.exp.speed.df.loc[id_exp, id_ant, :] = np.c_[np.around(np.sqrt(dx**2+dy**2) * fps, 3)]
                self.exp.speed_phi.df.loc[id_exp, id_ant, :] = np.c_[np.around(angle(list(zip(dx, dy)))*fps, 3)]

            self.exp.groupby('xy', [id_exp_name, id_ant_name], compute_speed4each_group)

            self.exp.write([name, name+'_x', name+'_y', name+'_phi'])

        if redo or redo_hist:
            self.exp.load(name)
            self.exp.hist1d(name_to_hist=name, result_name=hist_name,
                            bins=bins, label=hist_label, description=hist_description)
            self.exp.write(hist_name)

        else:
            self.exp.load(hist_name)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(yscale='log', normed=True)
        plotter.save(fig)

    def compute_mm10_speed(self):
        category = 'SpeedMM'
        time_window = 10

        name = 'speed'
        name_x = 'speed_x'
        name_y = 'speed_y'
        names = [name, name_x, name_y]

        self.exp.load(names)
        for name in names:
            result_name = self.exp.rolling_mean(
                name_to_average=name, window=time_window, category=category, is_angle=False)
            self.exp.write(result_name)

    def compute_mm20_speed(self, redo=False, redo_hist=False):
        name = 'speed'
        category = 'SpeedMM'
        time_window = 20
        result_name = 'mm' + str(time_window) + '_' + name
        hist_name = result_name + '_hist'
        bins = np.arange(0, 500, 1)
        hist_label = 'Distribution of the speed (mm/s), MM '+str(time_window)
        hist_description = 'Distribution of the instantaneous speed of the ants (mm/s)' \
                           ' smoothed with a moving mean of window length ' + str(time_window) + ' frames'
        if redo:
            self.exp.load(name)
            result_name = self.exp.rolling_mean(
                name_to_average=name, window=time_window, category=category, is_angle=False)
            self.exp.write(result_name)

        self.compute_hist(hist_name=hist_name, name=result_name, bins=bins,
                          hist_label=hist_label, hist_description=hist_description, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(yscale='log', normed=True)
        plotter.save(fig)

    def compute_mm1s_speed(self, redo=False, redo_hist=False):
        name = 'speed'
        category = 'SpeedMM'
        time_window = 100
        result_name = 'mm1s_' + name
        hist_name = result_name + '_hist'
        bins = np.arange(0, 500, 1)
        hist_label = 'Distribution of the speed (mm/s), MM 1s'
        hist_description = 'Distribution of the instantaneous speed of the ants (mm/s)' \
                           ' smoothed with a moving mean of window length 1 second'
        if redo:
            self.exp.load(name)
            self.exp.rolling_mean(
                name_to_average=name, window=time_window, result_name=result_name, category=category, is_angle=False)
            self.exp.write(result_name)

        self.compute_hist(hist_name=hist_name, name=result_name, bins=bins,
                          hist_label=hist_label, hist_description=hist_description, redo=redo, redo_hist=redo_hist)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
        fig, ax = plotter.plot(yscale='log', normed=True)
        plotter.save(fig)

    def compute_mm10_orientation(self):
        name = 'orientation'
        category = 'OrientationMM'
        time_window = 10

        self.exp.load(name)
        result_name = self.exp.rolling_mean(
            name_to_average=name, window=time_window, category=category, is_angle=True)

        self.exp.write(result_name)

    def compute_mm20_orientation(self):
        name = 'orientation'
        category = 'OrientationMM'
        time_window = 20

        self.exp.load(name)
        result_name = self.exp.rolling_mean(
            name_to_average=name, window=time_window, category=category, is_angle=True)

        self.exp.write(result_name)
