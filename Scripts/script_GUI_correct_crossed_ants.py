from __future__ import unicode_literals

import sys

import numpy as np
import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QSlider
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from matplotlib.path import Path

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from DataStructure.VariableNames import id_frame_name, id_ant_name, id_exp_name
from ExperimentGroups import ExperimentGroups
from Scripts.root import root
from Tools.MiscellaneousTools.Geometry import is_intersecting_df, distance


class MovieCanvas(FigureCanvas):

    def __init__(self, exp: ExperimentGroups, parent=None, id_exp=1, width=1920, height=1080*1.15):

        self.exp = exp
        self.id_exp = id_exp
        self.frame_width = width/200.
        self.frame_height = height/200.
        self.play = 0
        self.batch_length = 5
        self.dt = 50
        self.dpx = 60

        self.mode = 0
        self.event_text = 'crosses'

        self.data_manager = DataManager(exp, id_exp)

        self.fig, self.ax = self.init_figure()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.movie = self.exp.get_movie(id_exp)
        self.xy_df0 = self.data_manager.get_trajs()
        self.list_outside_ant = self.data_manager.get_outside_ant()
        self.not_crossing = {id_ant: [] for id_ant in self.list_outside_ant}

        self.events = None
        self.iter_event = 0

        self.iter_event = 0
        self.iter_focused_ant = 0
        self.frame = 0
        self.end_frame = 0

        self.frame_graph = None
        self.focused_xy_graph = None
        self.other_xy_graph = None
        self.candidates_graphs = []
        self.crossed_point_graph = None
        self.crossing_point_graph = None

        self.current_text = None
        self.clock_text = None

        self.events = None
        self.focused_ant_xy = None
        self.other_ant_xy = None

        self.x0 = 0
        self.y0 = 0
        self.dx = 0
        self.dy = 0

        self.next_ant()

        self.set_canvas_size()
        self.time_loop()

    def init_figure(self):
        fig = Figure(figsize=(self.frame_width, self.frame_height))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(0, 0.1, 1, 1)
        return fig, ax

    def prev_ant(self):
        print('prev ant')
        self.iter_event = 0
        if self.mode == 0:
            self.get_prev_other_ants(self.get_crossings)
        else:
            self.get_prev_other_ants(self.get_candidates)
        self.refresh()

    def get_prev_other_ants(self, get_events):

        self.events = []
        start_inter_ant = self.iter_focused_ant
        turn = False

        while len(self.events) == 0 and not turn:
            print(self.get_focused_ant_id())

            self.iter_focused_ant -= 1
            get_events()

            if self.iter_focused_ant < 0:
                self.iter_focused_ant = len(self.list_outside_ant) - 1
                break
            turn = start_inter_ant == self.iter_focused_ant

    def get_crossings(self):
        self.events = self.data_manager.get_crossing(
            self.get_focused_ant_id(), self.xy_df0, self.not_crossing[self.get_focused_ant_id()])

    def get_candidates(self):
        self.events = self.data_manager.search_for_traj_following(self.xy_df0, self.get_focused_ant_id())

    def get_candidates2(self):
        self.events = self.data_manager.search_for_traj_following2(self.xy_df0, self.get_focused_ant_id())

    def next_ant(self):
        self.iter_event = 0
        if self.mode == 0:
            self.get_next_other_ants(self.get_crossings)
        else:
            self.get_next_other_ants(self.get_candidates)
        self.refresh()

    def get_next_other_ants(self, get_events):

        self.events = []
        start_inter_ant = self.iter_focused_ant
        turn = False

        while len(self.events) == 0 and not turn:
            print(self.get_focused_ant_id())

            self.iter_focused_ant += 1
            get_events()

            if self.iter_focused_ant >= len(self.list_outside_ant):
                self.iter_focused_ant = 0
                break
            turn = start_inter_ant == self.iter_focused_ant

    def prev_event(self):
        if self.iter_event is None:
            self.ax.text(0, 0, 'No more '+self.event_text)
            self.draw()
        else:
            self.iter_event -= 1
            if self.iter_event < 0:
                self.iter_event = len(self.events) - 1

            self.refresh()

    def next_event(self):
        if self.iter_event is None:
            self.ax.text(0, 0, 'No more '+self.event_text)
            self.draw()
        else:
            self.iter_event += 1
            if self.iter_event >= len(self.events):
                self.iter_event = 0

            self.refresh()

    def not_a_cross(self):

        if self.mode == 0:
            focused_ant = self.get_focused_ant_id()
            other_ant = self.get_other_ant_id()
            frame = self.events[self.iter_event, 1]

            self.not_crossing[focused_ant].append((other_ant, frame))
            if self.get_other_ant_id() in self.list_outside_ant:
                self.not_crossing[self.get_other_ant_id()].append((focused_ant, frame))

            self.events = np.array(list(self.events[:self.iter_event]) + list(self.events[self.iter_event + 1:]))
            if len(self.events) == 0:
                self.iter_event = None
            elif self.iter_event == len(self.events)-1:
                self.iter_event = 0

            self.refresh()

    def search_for_candidates(self):
        self.mode = 1
        self.event_text = 'candidates'

        self.iter_event = 0
        id_ant = self.get_focused_ant_id()
        if self.data_manager.has_exit(self.xy_df0, id_ant):
            self.iter_event = None
        else:
            self.get_candidates()
        self.next_event()

    def search_for_candidates2(self):
        self.mode = 1
        self.event_text = 'candidates'

        self.iter_event = 0
        id_ant = self.get_focused_ant_id()
        if self.data_manager.has_exit(self.xy_df0, id_ant):
            self.iter_event = None
        else:
            self.get_candidates2()
        self.next_event()

    def search_for_crossings(self):
        self.mode = 0
        self.event_text = 'crosses'

        self.iter_event = 0
        self.get_crossings()
        self.next_event()

    def refresh(self):
        self.focused_ant_xy, self.other_ant_xy, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()
        self.reset_play()

    def decross(self):
        print('decrossing')
        focused_ant = self.get_focused_ant_id()
        crossing_ant = self.get_other_ant_id()
        crossing_frame = self.get_event_frame()

        self.xy_df0 = self.data_manager.decross(self.xy_df0, focused_ant, crossing_ant, crossing_frame)

        if self.mode == 0:
            self.search_for_crossings()
        else:
            self.search_for_candidates()

    def write(self):
        self.data_manager.write(self.xy_df0)

    def get_event_frame(self):
        if self.mode == 1 or self.iter_event is None or len(self.events) == 0:
            return self.xy_df0.loc[self.id_exp, self.get_focused_ant_id()].index.get_level_values(id_frame_name)[-1]
        else:
            return self.events[self.iter_event, 1]

    # def get_event_frame(self):
    #     return self.xy_df0.loc[self.id_exp, self.get_focused_ant(), :].index.get_level_values(id_frame_name)[-1]

    def get_focused_ant_id(self):
        return self.list_outside_ant[self.iter_focused_ant]

    def get_other_ant_id(self):
        return self.events[self.iter_event, 0]

    # def get_other_ant_id(self):
    #     return self.events[self.iter_event]

    def cropping_xy(self):
        focused_ant = self.get_focused_ant_id()
        frame = self.get_event_frame()

        xy = self.xy_df0.loc[self.id_exp, :, :].copy()
        focus_xy_df = xy.loc[focused_ant, :]
        focus_xy_df = focus_xy_df.loc[frame-self.dt:frame+self.dt]

        x0, y0 = int(np.mean(focus_xy_df.x)), int(np.mean(focus_xy_df.y))
        focus_xy_df.x -= x0
        focus_xy_df.y -= y0

        dx = int(max(np.nanmax(focus_xy_df.x), -np.nanmin(focus_xy_df.x)))
        dx = max(dx, self.dpx)

        dy = int(max(np.nanmax(focus_xy_df.y), -np.nanmin(focus_xy_df.y)))
        dy = max(dy, self.dpx)

        if len(self.events) == 0 or self.iter_event is None:
            other_xy_df = None
        else:
            other_ant = self.get_other_ant_id()

            other_xy_df = xy.loc[other_ant, :]
            other_xy_df = other_xy_df.loc[frame-self.dt:frame+self.dt]

            other_xy_df.x -= x0
            other_xy_df.y -= y0

            dx = max(dx, int(max(np.nanmax(other_xy_df.x), -np.nanmin(other_xy_df.x)))) + self.dpx
            dx = max(dx, self.dpx)

            dy = max(dy, int(max(np.nanmax(other_xy_df.y), -np.nanmin(other_xy_df.y)))) + self.dpx
            dy = max(dy, self.dpx)

            other_xy_df.x += dx
            other_xy_df.y += dy

        focus_xy_df.x += dx
        focus_xy_df.y += dy

        return focus_xy_df, other_xy_df, x0, y0, dx, dy

    def reset_play(self):
        if self.mode == 0:
            self.display_crossing()
        else:
            self.display_candidates()
        self.end_frame = self.frame+self.dt*2

    def display_crossing(self):
        if len(self.events) == 0 or self.iter_event is None:

            self.ax.cla()
            self.current_text = self.ax.text(
                0.5, 0.5, 'Ant ' + str(self.get_focused_ant_id()) + ': no crosses', color='black', weight='bold',
                size='xx-large', horizontalalignment='center', verticalalignment='top')
            self.draw()
        else:

            frame0 = self.focused_ant_xy.index[0]
            self.frame = frame0
            frame_img = self.crop_frame_img(self.movie.get_frame(frame0))

            self.ax.cla()
            self.frame_graph = self.ax.imshow(frame_img, cmap='gray')

            self.focused_xy_graph, = self.ax.plot(self.focused_ant_xy.x, self.focused_ant_xy.y, c='peru')
            self.other_xy_graph, = self.ax.plot(self.other_ant_xy.x, self.other_ant_xy.y, c='k')

            cross_frame = self.get_event_frame()
            self.crossing_point_graph, = self.ax.plot(
                self.focused_ant_xy.x.loc[cross_frame], self.focused_ant_xy.y.loc[cross_frame], 'o', c='darkred')
            self.crossed_point_graph, = self.ax.plot(
                self.other_ant_xy.x.loc[cross_frame], self.other_ant_xy.y.loc[cross_frame], 'o', c='darkred')

            id_ant = str(self.get_focused_ant_id())
            self.current_text = self.ax.text(
                self.dx, 0, 'ant ' + id_ant + ' (' + str(self.iter_event + 1) + '/' + str(len(self.events)) + ')',
                color='black', weight='bold', size='xx-large', horizontalalignment='center', verticalalignment='top')

            self.clock_text = self.ax.text(
                0, 0, str(frame0), color='green', weight='bold', size='xx-large',
                horizontalalignment='left', verticalalignment='top')

            self.ax.axis('equal')
            self.ax.axis('off')
            self.draw()

    def display_candidates(self):

        self.frame = self.get_event_frame() - self.dt
        frame_img = self.crop_frame_img(self.movie.get_frame(self.frame))

        self.ax.cla()
        self.frame_graph = self.ax.imshow(frame_img, cmap='gray')

        self.focused_xy_graph, = self.ax.plot(self.focused_ant_xy.x, self.focused_ant_xy.y, c='peru')

        if self.iter_event is None:
            self.current_text = self.ax.text(
                0.5, 0.5, 'Ant ' + str(self.get_focused_ant_id()) + ': has exited', color='black', weight='bold',
                size='xx-large', horizontalalignment='center', verticalalignment='top')
            self.draw()
            self.mode = 0
        elif len(self.events) == 0:
            self.current_text = self.ax.text(
                0.5, 0.5, 'Ant ' + str(self.get_focused_ant_id()) + ': no candidates', color='black', weight='bold',
                size='xx-large', horizontalalignment='center', verticalalignment='top')
            self.draw()
        else:

            self.other_xy_graph, = self.ax.plot(self.other_ant_xy.x, self.other_ant_xy.y, c='k')

            self.candidates_graphs = []
            for id_ant in self.events:
                xys = self.xy_df0.loc[self.id_exp, id_ant, self.frame:self.frame+self.dt*2]
                candidates_graph, = self.ax.plot(xys.x - self.x0 + self.dx, xys.y - self.y0 + self.dy, c='k')
                self.candidates_graphs.append(candidates_graph)

            id_ant = str(self.get_focused_ant_id())

            self.current_text = self.ax.text(
                self.dx, 0, 'ant ' + id_ant + ' (' + str(self.iter_event + 1)
                            + '/' + str(len(self.events)) + ')',
                color='black', weight='bold', size='xx-large', horizontalalignment='center',
                verticalalignment='top')

        self.clock_text = self.ax.text(
            0, 0, str(self.frame), color='green', weight='bold', size='xx-large',
            horizontalalignment='left', verticalalignment='top')

        self.ax.axis('equal')
        self.ax.axis('off')
        self.draw()

    def crop_frame_img(self, frame_img):
        return frame_img[self.y0 - self.dy:self.y0 + self.dy, self.x0 - self.dx:self.x0 + self.dx]

    def update_figure(self):
        if self.play == 1:
            self.frame += 1
            if self.frame == self.end_frame:
                self.play = -1
                self.reset_play()
            else:

                frame_img = self.crop_frame_img(self.movie.get_next_frame())
                self.frame_graph.set_data(frame_img)

                xy = self.focused_ant_xy.loc[self.frame - 20:self.frame]
                self.focused_xy_graph.set_xdata(xy.x)
                self.focused_xy_graph.set_ydata(xy.y)

                if self.other_ant_xy is not None:
                    xy = self.other_ant_xy.loc[self.frame - 20:self.frame]
                    self.other_xy_graph.set_xdata(xy.x)
                    self.other_xy_graph.set_ydata(xy.y)

                for candidate_graph in self.candidates_graphs:
                    candidate_graph.set_xdata([])
                    candidate_graph.set_ydata([])

                self.clock_text.set_text(self.frame)

                self.draw()
        elif self.play == -1:
            self.clock_text.set_text('stop')
            self.clock_text.set_color('red')
            self.play = 0
            self.draw()

    def set_to_play(self):
        self.play = 1

    def set_to_stop(self):
        self.play = 0

    def time_loop(self):
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start()

    def set_canvas_size(self):
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class DataManager:
    def __init__(self, exp: ExperimentGroups, id_exp):
        self.exp = exp
        self.id_exp = id_exp
        self.res_name_x = 'decrossed_x0'
        self.res_name_y = 'decrossed_y0'
        # self.exp.delete_data([self.res_name_x, self.res_name_y])
        self.name_xy = 'xy0'
        self.cross_thresh = 2
        self.gate_path = None
        self.mm2px = None
        self.search_distance = 10
        self.search_time_window = 10
        self.init()

    def init(self):
        self.exp.load(['traj_translation', 'entrance1', 'entrance2', 'mm2px'])
        self.mm2px = self.exp.get_value('mm2px', self.id_exp)

        xmin = min(self.exp.entrance1.df.loc[self.id_exp].x, self.exp.entrance2.df.loc[self.id_exp].x)
        ymin = min(self.exp.entrance1.df.loc[self.id_exp].y, self.exp.entrance2.df.loc[self.id_exp].y)
        ymax = max(self.exp.entrance1.df.loc[self.id_exp].y, self.exp.entrance2.df.loc[self.id_exp].y)
        xmax = max(self.exp.entrance1.df.loc[self.id_exp].x, self.exp.entrance2.df.loc[self.id_exp].x)
        dl = 20*self.mm2px

        gate1 = np.array([xmin - dl, ymin - dl])
        gate2 = np.array([xmin - dl, ymax + dl])
        gate3 = np.array([xmax + dl, ymax + dl])
        gate4 = np.array([xmax + dl, ymin - dl])

        self.gate_path = Path([gate1, gate2, gate3, gate4])

        if self.exp.is_name_existing(self.res_name_x):
            name_x = self.res_name_x
            name_y = self.res_name_y
            self.exp.load_as_2d(name_x, name_y, self.name_xy, 'x', 'y')
        else:
            name_x = 'interpolated_x0'
            name_y = 'interpolated_y0'
            self.exp.load_as_2d(name_x, name_y, self.name_xy, 'x', 'y')

    def get_trajs(self):
        xy_df = self.exp.get_df(self.name_xy).loc[self.id_exp, :, :]
        xy_df.x += np.array(self.exp.traj_translation.df.x.loc[self.id_exp])
        xy_df.y += np.array(self.exp.traj_translation.df.y.loc[self.id_exp])

        return xy_df

    def get_outside_ant(self):

        self.exp.load('from_outside')
        ant_df = self.exp.get_df('from_outside').loc[self.id_exp, :]
        ant_df = ant_df[ant_df == 1].dropna()
        list_outside_ant = list(ant_df.index.get_level_values(id_ant_name))

        return list_outside_ant

    def get_crossing(self, id_ant, xy_df, not_crossing):
        self.exp.load('mm2px')

        xy = xy_df.loc[self.id_exp, id_ant, :]
        focused_frame0 = xy.index.get_level_values(id_frame_name)[0]
        focused_frame1 = xy.index.get_level_values(id_frame_name)[-1]

        list_id_ant = list(set(xy_df.index.get_level_values(id_ant_name)))
        list_id_ant.remove(id_ant)

        res = []
        for id_ant2 in list_id_ant:
            xy2 = xy_df.loc[self.id_exp, id_ant2, :]
            other_frame0 = xy2.index.get_level_values(id_frame_name)[0]
            other_frame1 = xy2.index.get_level_values(id_frame_name)[-1]
            if focused_frame1 > other_frame0 or other_frame1 > focused_frame0:

                dframe = 10

                a = xy.iloc[:-dframe].copy()
                b = xy.iloc[dframe:].copy()
                b.index -= dframe

                c = xy2.iloc[:-dframe].copy()
                d = xy2.iloc[dframe:].copy()
                d.index -= dframe

                is_intersecting = is_intersecting_df(a, b, c, d)

                is_crossing = is_intersecting.dropna().astype(bool)
                is_crossing = is_crossing[is_crossing]

                if len(is_crossing) != 0:
                    cross_frame = []

                    for i in range(len(is_crossing)):

                        frame = is_crossing.index[i]
                        is_intersecting = [False]
                        dframe2 = 0

                        while not any(is_intersecting) and dframe2 <= dframe:
                            dframe2 += 1

                            a = xy.loc[frame:frame+dframe].copy()
                            b = xy.loc[frame+dframe2:frame+dframe+dframe2].copy()
                            b.index -= dframe2

                            c = xy2.loc[frame:frame+dframe].copy()
                            d = xy2.loc[frame+dframe2:frame+dframe+dframe2].copy()
                            d.index -= dframe2

                            is_intersecting = is_intersecting_df(a, b, c, d)

                        is_crossing2 = is_intersecting.dropna().astype(bool)
                        is_crossing2 = is_crossing2[is_crossing2]
                        frame = int(np.mean(is_crossing2.index))
                        if (id_ant2, frame) not in not_crossing and frame not in cross_frame:
                            cross_frame.append(frame)

                    crossing_ants = np.full(len(cross_frame), id_ant2)
                    res += (list(zip(crossing_ants, cross_frame)))

        res = np.array(res)
        if len(res) != 0:
            res = res[np.argsort(res[:, 1]), :]

        return res

    def decross(self, xy_df, crossed_ant, crossing_ant, cross_frame):
        exps = np.array(xy_df.index.get_level_values(id_exp_name))
        ants = np.array(xy_df.index.get_level_values(id_ant_name))
        frames = np.array(xy_df.index.get_level_values(id_frame_name))

        is_focused_exp = (exps == self.id_exp)
        is_crossed_ant = (ants == crossed_ant)
        is_crossing_ant = (ants == crossing_ant)
        is_after_frame = (frames > cross_frame)

        mask_crossed_ant = np.where(is_focused_exp*is_crossed_ant*is_after_frame)[0]
        mask_crossing_ant = np.where(is_focused_exp*is_crossing_ant*is_after_frame)[0]

        ants[mask_crossed_ant] = crossing_ant
        ants[mask_crossing_ant] = crossed_ant
        xy_df.reset_index(inplace=True)
        xy_df[id_ant_name] = ants
        xy_df.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)

        xy_df.sort_index(inplace=True)

        xy_df = self.__interpolate_time_series1d(xy_df, crossed_ant)
        xy_df = self.__interpolate_time_series1d(xy_df, crossing_ant)
        xy_df.sort_index(inplace=True)

        return xy_df

    def __interpolate_time_series1d(self, df: pd.DataFrame, id_ant):
        if id_ant in df.index.get_level_values(id_ant_name):
            df2 = df.loc[self.id_exp, id_ant]
            frames = np.array(df2.index.get_level_values(id_frame_name))
            dframes = frames[1:]-frames[:-1]

            hole_locations = np.where(dframes > 1)[0]
            if len(hole_locations) != 0:

                for i in range(0, len(hole_locations)):
                    frame0 = frames[hole_locations[i]]
                    frame1 = frames[hole_locations[i]+1]

                    val0 = df2.iloc[hole_locations[i]]
                    val1 = df2.iloc[hole_locations[i]+1]

                    num = frame1-frame0+1
                    val_range_x = np.around(np.linspace(val0.x, val1.x, num+2), 2)
                    val_range_y = np.around(np.linspace(val0.y, val1.y, num+2), 2)

                    for j, frame in enumerate(range(frame0+1, frame1)):
                        df.loc[(self.id_exp, id_ant, frame), 'x'] = val_range_x[j+1]
                        df.loc[(self.id_exp, id_ant, frame), 'y'] = val_range_y[j+1]

        return df

    def has_exit(self, xy_df, id_ant):
        xys = np.array(xy_df.loc[self.id_exp, id_ant, :])[-10:]
        has_exit = any(self.gate_path.contains_points(xys))
        return has_exit

    def search_for_traj_following(self, xy_df, id_ant):

        xys = xy_df.loc[self.id_exp, :, :]
        ants = list(set(np.array(xys.index.get_level_values(id_ant_name))))
        ants.remove(id_ant)

        ant_end_frame = np.max(np.array(xys.loc[id_ant, :].index.get_level_values(id_frame_name)))
        focused_ant_end_point = np.array(xys.loc[id_ant, ant_end_frame])

        res = []
        for id_ant2 in ants:
            ant2_start_frame = np.min(np.array(xys.loc[id_ant2, :].index.get_level_values(id_frame_name)))
            ant2_start_position = np.array(xys.loc[id_ant2, ant2_start_frame])
            if np.abs(ant2_start_frame-ant_end_frame) < self.search_time_window \
                    and distance(focused_ant_end_point, ant2_start_position)[0] < self.search_distance*self.mm2px:
                res.append([id_ant2, ant2_start_frame])

        return np.array(res)

    def search_for_traj_following2(self, xy_df, id_ant):

        xys = xy_df.loc[self.id_exp, :, :]
        ants = list(set(np.array(xys.index.get_level_values(id_ant_name))))
        ants.remove(id_ant)

        ant_end_frame = np.max(np.array(xys.loc[id_ant, :].index.get_level_values(id_frame_name)))

        focused_ant_end_position = np.array(xys.loc[id_ant, ant_end_frame])

        res = []
        for id_ant2 in ants:
            if (id_ant2, ant_end_frame) in xys.index:
                other_ant_end_position = np.array(xys.loc[id_ant2, ant_end_frame])

                if len(other_ant_end_position) != 0:
                    if distance(focused_ant_end_position, other_ant_end_position)[0] < self.search_distance*self.mm2px:

                        res.append([id_ant2, ant_end_frame])

        return np.array(res)

    def write(self, xy_df):
        df = self.exp.get_df(self.name_xy).copy()
        exps = np.array(df.index.get_level_values(id_exp_name))
        ants = np.array(df.index.get_level_values(id_ant_name))
        frames = np.array(df.index.get_level_values(id_frame_name))

        new_ants = np.array(xy_df.index.get_level_values(id_ant_name))
        new_frames = np.array(xy_df.index.get_level_values(id_frame_name))

        is_focused_exp = np.where(exps == self.id_exp)[0]
        ants[is_focused_exp] = new_ants
        frames[is_focused_exp] = new_frames

        df.index = pd.MultiIndex.from_tuples(
            list(zip(exps, ants, frames)), names=[id_exp_name, id_ant_name, id_frame_name])

        xy_df = xy_df - self.exp.get_df('traj_translation').loc[self.id_exp]
        df.loc[self.id_exp, :, :] = xy_df

        self.exp.add_new1d_from_df(df=pd.DataFrame(df.x), name=self.res_name_x,
                                   object_type='TimeSeries1d', category='CleanedRaw',
                                   label='decrossed x coordinate (px, in the cropped image system)',
                                   description='x coordinate in px and in the cropped image system,'
                                               'for which the trajectory of ant coming from outside has been manually '
                                               'corrected to have no crossing and is complete', replace=True)

        self.exp.add_new1d_from_df(df=pd.DataFrame(df.y), name=self.res_name_y,
                                   object_type='TimeSeries1d', category='CleanedRaw',
                                   label='decrossed y coordinate (px, in the cropped image system)',
                                   description='y coordinate in px and in the cropped image system,'
                                               'for which the trajectory of ant coming from outside has been manually '
                                               'corrected to have no crossing and is complete', replace=True)

        self.exp.write([self.res_name_x, self.res_name_y])


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, group, id_exp):

        self.bt_height = 30
        self.bt_length = 80
        self.dl = 5

        self.exp = ExperimentGroupBuilder(root).build(group)

        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.main_widget = QtWidgets.QWidget(self)
        self.movie_canvas = MovieCanvas(self.exp, self.main_widget, id_exp=id_exp)

        self.layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.layout.addWidget(self.movie_canvas)

        self.setWindowTitle("Experiment "+str(self.movie_canvas.id_exp))
        self.addToolBar(QtCore.Qt.BottomToolBarArea, NavigationToolbar(self.movie_canvas, self))

        self.button_group_box = QtWidgets.QGridLayout()
        self.add_all_buttons()
        self.add_group_box(self.button_group_box)

        self.slider_group_box = QtWidgets.QGridLayout()
        self.movie_time_window_slider = self.create_slider(
            0, 0, 'movie time window (frame)',
            min_val=40, max_val=1000, step=10, value=50, func=self.update_movie_time_window)
        self.search_time_window_slider = self.create_slider(
            0, 1, 'search time window (frame)',
            min_val=10, max_val=1000, step=10, value=50, func=self.update_search_time_window)
        self.search_distance_slider = self.create_slider(
            0, 2, 'search distance (mm)',
            min_val=5, max_val=500, step=1, value=10, func=self.update_search_distance)
        self.add_group_box(self.slider_group_box)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def add_all_buttons(self):
        self.create_button(1, 0, "Save", self.save)
        self.create_button(0, 0, "Quit", self.quit)

        self.create_button(1, 2, "Play", self.resume_play)
        self.create_button(1, 3, "RePlay", self.replay)
        self.create_button(0, 2, "Pause", self.pause_play)

        self.create_button(0, 5, "Prev ant", self.prev_ant)
        self.create_button(0, 6, "Next ant", self.next_ant)

        self.create_button(1, 5, "Prev event", self.prev_event)
        self.create_button(1, 6, "Next event", self.next_event)

        self.create_button(0, 8, "Decross", self.decross)
        self.create_button(1, 8, "Not crossing", self.not_a_crossing)

        self.create_button(0, 9, "Crossings", self.search_crossings)

        self.create_button(0, 10, "Candidate", self.search_candidate)
        self.create_button(1, 10, "Candidate2", self.search_candidate2)

    def add_group_box(self, layout):
        group_box = QtWidgets.QGroupBox()
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_Space:
            self.replay()
        elif key == Qt.Key_Right:
            self.next_event()
        elif key == Qt.Key_Left:
            self.prev_event()
        elif key == Qt.Key_Up:
            self.next_ant()
        elif key == Qt.Key_Down:
            self.prev_ant()
        elif key == Qt.Key_S:
            self.save()
        elif key == Qt.Key_Return:
            self.decross()
        elif key == Qt.Key_0:
            self.not_a_crossing()
        elif key == Qt.Key_C:
            self.search_candidate()
        else:
            pass
        self.update()

    def create_button(self, n_line, n_col, text, func):
        button = QPushButton(text)
        button.setFocusPolicy(QtCore.Qt.NoFocus)
        button.clicked.connect(func)
        self.button_group_box.addWidget(button, n_line, n_col)

    def create_slider(self, n_line, n_col, name, min_val, max_val, step, value, func):

        group_box = QtWidgets.QGroupBox(name)
        slider = QSlider(Qt.Horizontal)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setTickInterval(step)
        slider.setSingleStep(step)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(value)
        slider.setFocusPolicy(QtCore.Qt.NoFocus)
        slider.sliderReleased.connect(func)

        slider_box = QtWidgets.QVBoxLayout()
        slider_box.addWidget(slider)

        group_box.setLayout(slider_box)

        self.slider_group_box.addWidget(group_box, n_line, n_col)
        return slider

    def update_movie_time_window(self):
        self.movie_canvas.dt = self.movie_time_window_slider.value()
        self.movie_canvas.refresh()

    def update_search_time_window(self):
        self.movie_canvas.data_manager.search_time_window = self.search_time_window_slider.value()
        self.search_candidate()

    def update_search_distance(self):
        self.movie_canvas.data_manager.search_time_window = self.search_distance_slider.value()
        self.search_candidate()

    def replay(self):
        self.movie_canvas.reset_play()
        self.movie_canvas.set_to_play()

    def pause_play(self):
        self.movie_canvas.set_to_stop()

    def resume_play(self):
        self.movie_canvas.set_to_play()

    def prev_ant(self):
        self.movie_canvas.prev_ant()

    def next_ant(self):
        self.movie_canvas.next_ant()

    def prev_event(self):
        self.movie_canvas.prev_event()

    def next_event(self):
        self.movie_canvas.next_event()

    def decross(self):
        self.movie_canvas.decross()

    def not_a_crossing(self):
        self.movie_canvas.not_a_cross()

    def search_candidate(self):
        print('search candidate')
        self.movie_canvas.search_for_candidates()

    def search_candidate2(self):
        print('search candidate 2')
        self.movie_canvas.search_for_candidates2()

    def search_crossings(self):
        self.movie_canvas.search_for_crossings()

    def quit(self):
        self.save()
        self.close()

    def save(self):
        print('saving')
        self.movie_canvas.write()


qApp = QtWidgets.QApplication(sys.argv)

group0 = 'UO'

aw = ApplicationWindow(group0, id_exp=3)
aw.show()
sys.exit(qApp.exec_())
