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

        self.data_manager = DataManager(exp, id_exp)

        self.fig, self.ax = self.init_figure()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.movie = self.exp.get_movie(id_exp)
        self.xy_df0 = self.data_manager.get_trajs()
        self.list_outside_ant = self.data_manager.get_outside_ant()
        self.not_crossing = {id_ant: [] for id_ant in self.list_outside_ant}

        self.list_end_candidates = None
        self.iter_candidate = 0

        self.iter_cross = 0
        self.iter_ant = 0
        self.frame = 0

        self.frame_graph = None
        self.crossed_xy_graph = None
        self.crossing_xy_graph = None
        self.candidate_graph = None
        self.crossed_point_graph = None
        self.crossing_point_graph = None

        self.current_text = None
        self.clock_text = None

        self.crossing = None
        self.crossed_xy = None
        self.crossing_xy = None

        self.x0 = 0
        self.y0 = 0
        self.dx = 0
        self.dy = 0

        self.next_ant()

        self.set_canvas_size()
        self.time_loop()

    def prev_ant(self):
        print('prev ant')
        if self.iter_ant is None:
            self.ax.text(0, 0, 'No more ants')
            self.draw()
        else:
            self.iter_cross = 0
            self.iter_ant -= 1
            if self.iter_ant < 0:
                self.iter_ant = len(self.list_outside_ant)-1
            self.crossing = self.data_manager.get_crossing(
                self.get_focused_ant(), self.xy_df0, self.not_crossing[self.get_focused_ant()])

            start_inter_ant = self.iter_ant
            self.iter_ant -= 1
            while len(self.crossing) == 0 and self.iter_ant != start_inter_ant:
                print(self.get_focused_ant())
                if self.iter_ant < 0:
                    self.iter_ant = len(self.list_outside_ant)
                self.iter_ant -= 1

            self.crossing = self.data_manager.get_crossing(
                self.get_focused_ant(), self.xy_df0, self.not_crossing[self.get_focused_ant()])

            self.refresh()

    def next_ant(self):
        print('next ant')
        if self.iter_ant is None:
            self.ax.text(0, 0, 'No more ants')
            self.draw()
        else:
            self.iter_cross = 0
            self.iter_ant += 1
            if self.iter_ant >= len(self.list_outside_ant):
                self.iter_ant = 0
            self.crossing = self.data_manager.get_crossing(
                self.get_focused_ant(), self.xy_df0, self.not_crossing[self.get_focused_ant()])

            start_inter_ant = self.iter_ant
            self.iter_ant += 1
            while len(self.crossing) == 0 and self.iter_ant != start_inter_ant:
                print(self.get_focused_ant())
                if self.iter_ant >= len(self.list_outside_ant):
                    self.iter_ant = -1
                self.iter_ant += 1
                self.crossing = self.data_manager.get_crossing(
                    self.get_focused_ant(), self.xy_df0, self.not_crossing[self.get_focused_ant()])

            self.refresh()

    def prev_cross(self):
        if self.iter_cross is None:
            self.ax.text(0, 0, 'No more crosses')
            self.draw()
        else:
            self.iter_cross -= 1
            if self.iter_cross < 0:
                self.iter_cross = len(self.crossing)-1

            self.refresh()

    def next_cross(self):
        if self.iter_cross is None:
            self.ax.text(0, 0, 'No more crosses')
            self.draw()
        else:
            self.iter_cross += 1
            if self.iter_cross >= len(self.crossing):
                self.iter_cross = 0

            self.refresh()

    def not_a_cross(self):
        self.not_crossing[self.get_focused_ant()].append(tuple(self.crossing[self.iter_cross, :]))
        if self.get_crossing_ant() in self.list_outside_ant:
            self.not_crossing[self.get_crossing_ant()].append(tuple(self.crossing[self.iter_cross, :]))
        self.crossing = np.array(list(self.crossing[:self.iter_cross])+list(self.crossing[self.iter_cross+1:]))
        if len(self.crossing) == 0:
            self.iter_cross = None
        elif self.iter_cross == len(self.crossing)-1:
            self.iter_cross = 0
        # self.next_cross()
        self.refresh()

    def search_for_candidate(self):
        self.mode = 1
        id_ant = self.get_focused_ant()
        if self.data_manager.has_exit(self.xy_df0, id_ant):
            self.list_end_candidates = None
        else:
            self.list_end_candidates = self.data_manager.search_for_traj_following(self.xy_df0, id_ant)
        self.display_candidates()

    def next_candidate(self):
        self.iter_candidate += 1
        if self.iter_candidate >= len(self.list_end_candidates):
            self.iter_candidate = 0

        self.refresh()

    def prev_candidate(self):
        self.iter_candidate -= 1
        if self.iter_candidate < 0:
            self.iter_candidate = len(self.list_end_candidates)-1

        self.refresh()

    def display_candidates(self):

        if self.list_end_candidates is None:
            self.ax.cla()
            self.current_text = self.ax.text(
                0.5, 0.5, 'Ant ' + str(self.get_focused_ant()) + ': has exited', color='black', weight='bold',
                size='xx-large', horizontalalignment='center', verticalalignment='top')
            self.draw()
            self.mode = 0
        elif len(self.list_end_candidates) == 0:
            self.ax.cla()
            self.current_text = self.ax.text(
                0.5, 0.5, 'Ant ' + str(self.get_focused_ant()) + ': no candidates', color='black', weight='bold',
                size='xx-large', horizontalalignment='center', verticalalignment='top')
            self.draw()
        else:

            frame0 = self.crossed_xy.index[0]
            self.frame = frame0
            frame_img = self.crop_frame_img(self.movie.get_frame(frame0))

            self.ax.cla()
            self.frame_graph = self.ax.imshow(frame_img, cmap='gray')

            self.crossed_xy_graph, = self.ax.plot(self.crossed_xy.x, self.crossed_xy.y, c='peru')
            self.candidate_graph = []

            for id_ant in self.list_end_candidates:
                xys = self.xy_df.loc[id_ant, :]
                self.crossed_xy_graph, = self.ax.plot(xys.x, xys.y, c='peru')

            id_ant = str(self.get_focused_ant())
            self.current_text = self.ax.text(
                self.dx, 0, 'ant ' + id_ant + ' (' + str(self.iter_cross) + '/' + str(len(self.crossing)) + ')',
                color='black', weight='bold', size='xx-large', horizontalalignment='center',
                verticalalignment='top')

            self.clock_text = self.ax.text(
                0, 0, str(frame0), color='green', weight='bold', size='xx-large',
                horizontalalignment='left', verticalalignment='top')

            self.ax.axis('equal')
            self.ax.axis('off')
            self.draw()

    def refresh(self):
        self.crossed_xy, self.crossing_xy, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()
        self.reset_play()

    def decross(self):
        print('decrossing')
        crossed_ant = self.get_focused_ant()
        crossing_ant = self.get_crossing_ant()
        cross_frame = self.get_cross_frame()

        self.xy_df0 = self.data_manager.decross(self.xy_df0, crossed_ant, crossing_ant, cross_frame)
        self.crossing = self.data_manager.get_crossing(
            self.get_focused_ant(), self.xy_df0, self.not_crossing[self.get_focused_ant()])
        self.iter_cross = 0
        self.refresh()

    def write(self):
        self.data_manager.write(self.xy_df0)

    def get_cross_frame(self):
        return self.crossing[self.iter_cross, 1]

    def get_end_frame(self):
        return self.xy_df0.loc[self.get_crossing_ant(), :].iloc[-1]

    def get_focused_ant(self):
        return self.list_outside_ant[self.iter_ant]

    def get_crossing_ant(self):
        return self.crossing[self.iter_cross, 0]

    def get_candidate_ant(self):
        return self.crossing[self.iter_cross, 0]

    def init_figure(self):
        fig = Figure(figsize=(self.frame_width, self.frame_height))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(0, 0.1, 1, 1)
        return fig, ax

    def time_loop(self):
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start()

    def set_canvas_size(self):
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def cropping_xy(self):
        if len(self.crossing) == 0 or self.iter_cross is None:
            return None, None, None, None, None, None
        else:
            focus_ant = self.get_focused_ant()

            if self.mode == 0:
                other_ant = self.get_crossing_ant()
                frame = self.get_cross_frame()
            else:
                other_ant = self.get_crossing_ant()
                frame = self.get_end_frame()

            xy = self.xy_df0.loc[self.id_exp, :, :].copy()
            focus_xy_df = xy.loc[focus_ant, :]
            focus_xy_df = focus_xy_df.loc[frame-self.dt:frame+self.dt]
            other_xy_df = xy.loc[other_ant, :]
            other_xy_df = other_xy_df.loc[frame-self.dt:frame+self.dt]

            x0, y0 = int(np.mean(focus_xy_df.x)), int(np.mean(focus_xy_df.y))
            focus_xy_df.x -= x0
            focus_xy_df.y -= y0
            other_xy_df.x -= x0
            other_xy_df.y -= y0

            dx = max(np.nanmax(focus_xy_df.x), -np.nanmin(focus_xy_df.x))
            dx = int(max(dx, max(np.nanmax(other_xy_df.x), -np.nanmin(other_xy_df.x)))) + self.dpx
            dx = max(dx, self.dpx)

            dy = max(np.nanmax(focus_xy_df.y), -np.nanmin(focus_xy_df.y))
            dy = int(max(dy, max(np.nanmax(other_xy_df.y), -np.nanmin(other_xy_df.y)))) + self.dpx
            dy = max(dy, self.dpx)

            focus_xy_df.x += dx
            focus_xy_df.y += dy
            other_xy_df.x += dx
            other_xy_df.y += dy

            return focus_xy_df, other_xy_df, x0, y0, dx, dy

    def reset_play(self, non_text='no crosses'):
        if len(self.crossing) == 0 or self.iter_cross is None:

            self.ax.cla()
            self.current_text = self.ax.text(
                0.5, 0.5, 'Ant ' + str(self.get_focused_ant()) + ': ' + non_text, color='black', weight='bold',
                size='xx-large', horizontalalignment='center', verticalalignment='top')
            self.draw()
        else:

            frame0 = self.crossed_xy.index[0]
            self.frame = frame0
            frame_img = self.crop_frame_img(self.movie.get_frame(frame0))

            self.ax.cla()
            self.frame_graph = self.ax.imshow(frame_img, cmap='gray')

            self.crossed_xy_graph, = self.ax.plot(self.crossed_xy.x, self.crossed_xy.y, c='peru')
            self.crossing_xy_graph, = self.ax.plot(self.crossing_xy.x, self.crossing_xy.y, c='k')

            cross_frame = self.get_cross_frame()
            self.crossed_point_graph, = self.ax.plot(
                self.crossed_xy.x.loc[cross_frame], self.crossed_xy.y.loc[cross_frame], 'o', c='darkred')
            self.crossing_point_graph, = self.ax.plot(
                self.crossing_xy.x.loc[cross_frame], self.crossing_xy.y.loc[cross_frame], 'o', c='darkred')

            id_ant = str(self.get_focused_ant())
            self.current_text = self.ax.text(
                self.dx, 0, 'ant '+id_ant + ' ('+str(self.iter_cross)+'/'+str(len(self.crossing))+')',
                color='black', weight='bold', size='xx-large', horizontalalignment='center', verticalalignment='top')

            self.clock_text = self.ax.text(
                0, 0, str(frame0), color='green', weight='bold', size='xx-large',
                horizontalalignment='left', verticalalignment='top')

            self.ax.axis('equal')
            self.ax.axis('off')
            self.draw()

    def crop_frame_img(self, frame_img):
        return frame_img[self.y0 - self.dy:self.y0 + self.dy, self.x0 - self.dx:self.x0 + self.dx]

    def update_figure(self):
        if self.play == 1:
            self.frame += 1
            if self.frame >= self.crossed_xy.index[-1]:
                self.play = -1
                self.reset_play()
            else:

                frame_img = self.crop_frame_img(self.movie.get_next_frame())
                self.frame_graph.set_data(frame_img)

                xy = self.crossed_xy.loc[self.frame - 20:self.frame]
                self.crossed_xy_graph.set_xdata(xy.x)
                self.crossed_xy_graph.set_ydata(xy.y)

                xy = self.crossing_xy.loc[self.frame - 20:self.frame]
                self.crossing_xy_graph.set_xdata(xy.x)
                self.crossing_xy_graph.set_ydata(xy.y)

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


class DataManager:
    def __init__(self, exp: ExperimentGroups, id_exp):
        self.exp = exp
        self.id_exp = id_exp
        self.res_name_x = 'decrossed_x0'
        self.res_name_y = 'decrossed_y0'
        self.name_xy = 'xy0'
        self.cross_thresh = 2
        self.gate_path = None
        self.dist_end = 10
        self.mm2px = None
        self.init()

    def init(self):
        self.exp.load(['traj_translation', 'entrance1', 'entrance2', 'mm2px'])
        self.mm2px = self.exp.get_value('mm2px', self.id_exp)

        xmin = min(self.exp.entrance1.df.loc[self.id_exp].x, self.exp.entrance2.df.loc[self.id_exp].x)
        ymin = min(self.exp.entrance1.df.loc[self.id_exp].y, self.exp.entrance2.df.loc[self.id_exp].y)
        ymax = max(self.exp.entrance1.df.loc[self.id_exp].y, self.exp.entrance2.df.loc[self.id_exp].y)
        xmax = max(self.exp.entrance1.df.loc[self.id_exp].x, self.exp.entrance2.df.loc[self.id_exp].x)
        dl = 50*self.mm2px
        gate1 = [xmin - dl, ymin - dl]
        gate2 = [xmin - dl, ymax + dl]
        gate3 = [xmax + dl, ymax + dl]
        gate4 = [xmax + dl, ymin - dl]

        self.gate_path = Path([gate1, gate2, gate3, gate4])

        if self.exp.is_name_existing(self.res_name_x):
            name_x = self.res_name_x
            name_y = self.res_name_y
            self.exp.load_as_2d(name_x, name_y, self.name_xy, 'x', 'y')
            self.exp.operation_between_2names(self.name_xy, 'traj_translation', lambda x, y: x+y, 'x', 'x')
            self.exp.operation_between_2names(self.name_xy, 'traj_translation', lambda x, y: x+y, 'y', 'y')
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
        list_id_ant = list(set(xy_df.index.get_level_values(id_ant_name)))
        list_id_ant.remove(id_ant)

        res = []
        for id_ant2 in list_id_ant:
            xy2 = xy_df.loc[self.id_exp, id_ant2, :]

            dframe = 10
            a = xy.iloc[:-dframe].copy()
            a.index += dframe
            b = xy.iloc[dframe:].copy()
            c = xy2.iloc[:-dframe].copy()
            c.index += dframe
            d = xy2.iloc[dframe:].copy()
            is_intersecting = is_intersecting_df(a, b, c, d)

            is_crossing = is_intersecting.dropna().astype(bool)
            is_crossing = is_crossing[is_crossing]

            frame_when_cross_is_happening = np.array(is_crossing.index) + dframe/2
            if len(frame_when_cross_is_happening) != 0:
                dframe_when_cross_is_happening = frame_when_cross_is_happening[1:]-frame_when_cross_is_happening[:-1]

                between_crosses = list(np.where(dframe_when_cross_is_happening > self.cross_thresh)[0])
                between_crosses = [0]+between_crosses+[len(dframe_when_cross_is_happening)]

                cross_frame = []
                for i in range(1, len(between_crosses)):
                    frames = frame_when_cross_is_happening[between_crosses[i-1]:between_crosses[i]+1]
                    middle_frame_cross = int(frames[int(len(frames)/2)])
                    if (id_ant2, middle_frame_cross) not in not_crossing:
                        cross_frame.append(middle_frame_cross)

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
        is_after_frame = (frames >= cross_frame)

        mask_crossed_ant = np.where(is_focused_exp*is_crossed_ant*is_after_frame)[0]
        mask_crossing_ant = np.where(is_focused_exp*is_crossing_ant*is_after_frame)[0]

        ants[mask_crossed_ant] = crossing_ant
        ants[mask_crossing_ant] = crossed_ant
        xy_df.reset_index(inplace=True)
        xy_df[id_ant_name] = ants
        xy_df.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)

        xy_df.sort_index(inplace=True)

        return xy_df

    def has_exit(self, xy_df, id_ant):
        xys = np.array(xy_df.loc[self.id_exp, id_ant, :])[-10:]
        has_exit = any(self.gate_path.contains_points(xys))
        return has_exit

    def search_for_traj_following(self, xy_df, id_ant):

        xys = xy_df.loc[self.id_exp, :, :]
        ants = list(np.array(xys.index.get_level_values(id_ant_name)))
        ants.remove(id_ant)

        focused_ant_end_point = np.array(xys.loc[id_ant, :].iloc[-1])

        res = []
        for id_ant2 in ants:
            ant_end_point = np.array(xys.loc[id_ant2, :].iloc[-1])
            if distance(focused_ant_end_point, ant_end_point) < self.dist_end*self.mm2px:
                res.append(id_ant2)

        return res

    def write(self, xy_df):

        df_x = self.exp.get_df(self.name_xy).x
        df_x.loc[self.id_exp, :, :].x = xy_df.x
        self.exp.add_new1d_from_df(df=pd.DataFrame(df_x), name=self.res_name_x,
                                   object_type='TimeSeries1d', category='CleanedRaw',
                                   label='decrossed x coordinate (px, in the cropped image system)',
                                   description='x coordinate in px and in the cropped image system,'
                                               'for which the trajectory of ant coming from outside has been manually '
                                               'corrected to have no crossing and is complete', replace=True)

        df_y = self.exp.get_df(self.name_xy).y
        df_y.loc[self.id_exp, :, :].y = xy_df.y
        self.exp.add_new1d_from_df(df=pd.DataFrame(df_y), name=self.res_name_y,
                                   object_type='TimeSeries1d', category='CleanedRaw',
                                   label='decrossed y coordinate (px, in the cropped image system)',
                                   description='y coordinate in px and in the cropped image system,'
                                               'for which the trajectory of ant coming from outside has been manually '
                                               'corrected to have no crossing and is complete', replace=True)

        self.exp.operation_between_2names(self.res_name_x, 'traj_translation', lambda x, y: x-y, 'x', 'x')
        self.exp.operation_between_2names(self.res_name_y, 'traj_translation', lambda x, y: x-y, 'y', 'y')

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
            0, 1, 'movie time window', min_val=40, max_val=1000, step=10, value=50, func=self.update_movie_time_window)
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

        self.create_button(1, 5, "Prev cross", self.prev_cross)
        self.create_button(1, 6, "Next cross", self.next_cross)

        self.create_button(0, 8, "Decross", self.decross)
        self.create_button(1, 8, "Not crossing", self.not_a_crossing)

        self.create_button(0, 9, "Candidate", self.search_candidate)

    def add_group_box(self, layout):
        group_box = QtWidgets.QGroupBox()
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_Space:
            self.replay()
        elif key == Qt.Key_Right:
            self.next_cross()
        elif key == Qt.Key_Left:
            self.prev_cross()
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

    def prev_cross(self):
        self.movie_canvas.prev_cross()

    def next_cross(self):
        self.movie_canvas.next_cross()

    def decross(self):
        self.movie_canvas.decross()

    def not_a_crossing(self):
        self.movie_canvas.not_a_cross()

    def search_candidate(self):
        self.movie_canvas.search_for_candidate()

    def quit(self):
        self.movie_canvas.write()
        self.close()

    def save(self):
        self.movie_canvas.write()


qApp = QtWidgets.QApplication(sys.argv)

group0 = 'UO'

aw = ApplicationWindow(group0, id_exp=7)
aw.show()
sys.exit(qApp.exec_())
