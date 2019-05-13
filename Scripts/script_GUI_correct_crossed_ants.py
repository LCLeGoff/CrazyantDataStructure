from __future__ import unicode_literals

import sys

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from DataStructure.VariableNames import id_frame_name, id_ant_name, id_exp_name
from ExperimentGroups import ExperimentGroups
from Scripts.root import root
from Tools.MiscellaneousTools.Geometry import distance_df


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

        self.data_manager = DataManager(exp, id_exp)

        self.fig, self.ax = self.init_figure()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.movie = self.exp.get_movie(id_exp)
        self.xy_df0 = self.data_manager.get_trajs()
        self.list_outside_ant = self.data_manager.get_outside_ant()

        self.iter_cross = 0
        self.iter_ant = 0
        self.frame = 0

        self.crossing = None
        self.frame_graph = None
        self.crossed_xy_graph = None
        self.crossing_xy_graph = None
        self.current_text = None
        self.clock_text = None
        self.crossed_xy = None
        self.crossing_xy = None
        self.crossed_graph = None
        self.crossing_graph = None
        self.x0 = 0
        self.y0 = 0
        self.dx = 0
        self.dy = 0

        self.next_ant()

        self.set_canvas_size()
        self.time_loop()

    def prev_ant(self):
        if self.iter_ant is None:
            self.ax.text(0, 0, 'No more ants')
            self.draw()
        else:
            self.iter_cross = 0
            self.iter_ant -= 1
            if self.iter_ant < 0:
                self.iter_ant = len(self.list_outside_ant)-1

            self.crossing = self.data_manager.get_crossing(self.get_crossed_ant(), self.xy_df0)
            self.refresh()

    def next_ant(self):
        if self.iter_ant is None:
            self.ax.text(0, 0, 'No more ants')
            self.draw()
        else:
            self.iter_cross = 0
            self.iter_ant += 1
            if self.iter_ant >= len(self.list_outside_ant):
                self.iter_ant = 0

            self.crossing = self.data_manager.get_crossing(self.get_crossed_ant(), self.xy_df0)
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

    def refresh(self):
        self.crossed_xy, self.crossing_xy, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()
        self.reset_play()

    def decross(self):
        print('decrossing')
        crossed_ant = self.get_crossed_ant()
        crossing_ant = self.get_crossing_ant()
        cross_frame = self.get_cross_frame()

        self.xy_df0 = self.data_manager.decross(self.xy_df0, crossed_ant, crossing_ant, cross_frame)
        self.crossing = self.data_manager.get_crossing(self.get_crossed_ant(), self.xy_df0)
        self.refresh()

    def write(self):
        self.data_manager.write(self.xy_df0)

    def get_cross_frame(self):
        return self.crossing[self.iter_cross, 1]

    def get_crossed_ant(self):
        return self.list_outside_ant[self.iter_ant]

    def get_crossing_ant(self):
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
            crossed_ant = self.get_crossed_ant()
            crossing_ant = self.get_crossing_ant()
            frame = self.get_cross_frame()

            xy = self.xy_df0.loc[self.id_exp, :, :].copy()
            crossed_xy_df = xy.loc[crossed_ant, :]
            crossed_xy_df = crossed_xy_df.loc[frame-self.dt:frame+self.dt]
            crossing_xy_df = xy.loc[crossing_ant, :]
            crossing_xy_df = crossing_xy_df.loc[frame-self.dt:frame+self.dt]

            x0, y0 = int(np.mean(crossed_xy_df.x)), int(np.mean(crossed_xy_df.y))
            crossed_xy_df.x -= x0
            crossed_xy_df.y -= y0
            crossing_xy_df.x -= x0
            crossing_xy_df.y -= y0

            dx = max(np.nanmax(crossed_xy_df.x), -np.nanmin(crossed_xy_df.x))
            dx = int(max(dx, max(np.nanmax(crossing_xy_df.x), -np.nanmin(crossing_xy_df.x)))) + self.dpx
            dx = max(dx, self.dpx)

            dy = max(np.nanmax(crossed_xy_df.y), -np.nanmin(crossed_xy_df.y))
            dy = int(max(dy, max(np.nanmax(crossing_xy_df.y), -np.nanmin(crossing_xy_df.y)))) + self.dpx
            dy = max(dy, self.dpx)

            crossed_xy_df.x += dx
            crossed_xy_df.y += dy
            crossing_xy_df.x += dx
            crossing_xy_df.y += dy

            return crossed_xy_df, crossing_xy_df, x0, y0, dx, dy

    def reset_play(self):
        if len(self.crossing) == 0 or self.iter_cross is None:

            self.ax.cla()
            self.current_text = self.ax.text(
                0.5, 0.5, 'No crosses', color='black', weight='bold',
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
            self.crossed_graph, = self.ax.plot(
                self.crossed_xy.x.loc[cross_frame], self.crossed_xy.y.loc[cross_frame], 'o', c='darkred')
            self.crossing_graph, = self.ax.plot(
                self.crossing_xy.x.loc[cross_frame], self.crossing_xy.y.loc[cross_frame], 'o', c='darkred')

            id_ant = str(self.get_crossed_ant())
            self.current_text = self.ax.text(
                self.dx, 0, 'ant '+id_ant + ' ('+str(self.iter_cross)+'/'+str(len(self.crossing))+')',
                color='black', weight='bold', size='xx-large', horizontalalignment='center', verticalalignment='top')

            self.clock_text = self.ax.text(
                0, 0, 'Stop', color='green', weight='bold', size='xx-large',
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
        self.min_dist_in_mm = 4
        self.cross_thresh = 2
        self.init()

    def init(self):
        if self.exp.is_name_existing(self.res_name_x):
            name_x = self.res_name_x
            name_y = self.res_name_y
        else:
            name_x = 'x0'
            name_y = 'y0'
        self.exp.load_as_2d(name_x, name_y, self.name_xy, 'x', 'y')
        self.exp.load('traj_translation')

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

    def get_crossing(self, id_ant, xy_df):
        self.exp.load('mm2px')
        min_dist = self.min_dist_in_mm*self.exp.get_value('mm2px', self.id_exp)

        xy = xy_df.loc[self.id_exp, id_ant, :]
        list_id_ant = list(set(xy_df.index.get_level_values(id_ant_name)))
        list_id_ant.remove(id_ant)

        res = []
        for id_ant2 in list_id_ant:
            xy2 = xy_df.loc[self.id_exp, id_ant2, :]

            distances = distance_df(xy, xy2)
            distances = distances[distances < min_dist].dropna()
            frame_when_cross_is_happening = np.array(distances.index.get_level_values(id_frame_name))
            if len(frame_when_cross_is_happening) != 0:
                dframe_when_cross_is_happening = frame_when_cross_is_happening[1:]-frame_when_cross_is_happening[:-1]

                between_crosses = list(np.where(dframe_when_cross_is_happening > self.cross_thresh)[0])
                between_crosses = [0]+between_crosses+[len(dframe_when_cross_is_happening)]

                cross_frame = []
                for i in range(1, len(between_crosses)):
                    frames = frame_when_cross_is_happening[between_crosses[i-1]:between_crosses[i]+1]
                    middle_frame_cross = frames[int(len(frames)/2)]
                    cross_frame.append(middle_frame_cross)

                crossing_ants = np.full(len(cross_frame), id_ant2)
                res += (list(zip(crossing_ants, cross_frame)))

        return np.array(res)

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

    def write(self, xy_df):

        df_x = self.exp.get_df(self.name_xy).x
        df_x.loc[self.id_exp, :, :] = xy_df
        self.exp.add_new1d_from_df(df=df_x, name=self.res_name_x, object_type='TimeSeries1d', category='Raw',
                                   label='decrossed x coordinate (px, in the cropped image system)',
                                   description='x coordinate in px and in the cropped image system,'
                                               'for which the trajectory of ant coming from outside has been manually '
                                               'corrected to have no crossing and is complete')

        df_y = self.exp.get_df(self.name_xy).y
        df_y.loc[self.id_exp, :, :] = xy_df
        self.exp.add_new1d_from_df(df=df_y, name=self.res_name_y, object_type='TimeSeries1d', category='Raw',
                                   label='decrossed y coordinate (px, in the cropped image system)',
                                   description='y coordinate in px and in the cropped image system,'
                                               'for which the trajectory of ant coming from outside has been manually '
                                               'corrected to have no crossing and is complete')

        self.exp.operation_between_2names(self.res_name_x, 'traj_translation', lambda x, y: x-y)
        self.exp.operation_between_2names(self.res_name_y, 'traj_translation', lambda x, y: x-y)

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

        layout = QtWidgets.QVBoxLayout(self.main_widget)
        layout.addWidget(self.movie_canvas)

        self.setWindowTitle("Experiment"+str(self.movie_canvas.id_exp))
        self.addToolBar(QtCore.Qt.BottomToolBarArea, NavigationToolbar(self.movie_canvas, self))

        self.create_button(1, 0, "Save", self.save)
        self.create_button(0, 0, "Quit", self.quit)

        self.create_button(1, 2, "Play", self.resume_play)
        self.create_button(1, 3, "RePlay", self.replay)
        self.create_button(0, 2.5, "Pause", self.pause_play)

        self.create_button(0, 5, "Prev ant", self.prev_ant)
        self.create_button(0, 6, "Next ant", self.next_ant)

        self.create_button(1, 5, "Prev cross", self.prev_ant)
        self.create_button(1, 6, "Next cross", self.next_ant)

        self.create_button(0, 8, "Decross", self.decross)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

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
        else:
            pass
        self.update()

    def create_button(self, n_line, n_col, text, func):
        button = QPushButton(text, self)
        button.setGeometry(
            QRect(
                10 + (self.bt_length+self.dl) * n_col,
                560 + (self.bt_height + self.dl) * n_line,
                self.bt_length, self.bt_height))
        button.clicked.connect(func)

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
