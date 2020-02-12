from __future__ import unicode_literals

import sys

import numpy as np
import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QSlider
from matplotlib import colors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from scipy.signal import argrelextrema

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from DataStructure.VariableNames import id_frame_name, id_ant_name
from ExperimentGroups import ExperimentGroups
from Scripts.root import root
# from Tools.MiscellaneousTools.Geometry import angle_distance_df, norm_angle_tab


class MovieCanvas(FigureCanvas):

    def __init__(self, exp: ExperimentGroups, outside=True, parent=None, id_exp=1):

        self.exp = exp
        self.id_exp = id_exp
        self.frame_width = 1920
        self.frame_height = 1080
        self.zoom = 4
        self.play = 0
        self.batch_length1 = 0.5
        self.batch_length2 = 0.5
        self.outside = outside

        self.exp.load(['mm2px', 'fps'], reload=False)
        self.fps = self.exp.get_value('fps', id_exp)

        self.fig, self.ax_ant, self.ax_speed, self.ax_body_orient, self.ax_orient, self.ax_diff = self.init_figure()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.xy_df0, self.speed_df0, self.body_orient_df0, self.orient_df0, self.diff_df0,\
            self.list_id_ants, self.movie = self.get_traj_and_movie()

        self.iter_id_ant = 0
        self.id_ant = self.get_id_ant()
        self.list_frame_batch = self.get_frame_batches()
        self.speed_line, self.body_orient_line, self.orient_line, self.diff_line = self.get_linearization()

        self.frame_graph = None
        self.xy_graph = None
        self.speed_line_graph = None
        self.body_orient_line_graph = None
        self.orient_line_graph = None
        self.diff_line_graph = None
        self.batch_text = None
        self.clock_text = None

        self.iter_frame = 0
        self.iter_frame_batch = -1

        self.next_frame_batch()

        self.frame_batch = self.get_frame_batch()
        self.xy_df, self.speed_df, self.body_orient_df, self.orient_df, self.diff_df, \
            self.color_df, self.size_df, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()

        self.reset_play()

        self.set_canvas_size()
        self.time_loop()

    def get_traj_and_movie(self):
        fps = self.exp.get_value('fps', self.id_exp)

        self.exp.load_as_2d('mm10_x', 'mm10_y', 'xy', 'x', 'y', replace=True, reload=False)
        speed_name = 'speed'
        # body_orient_name = 'mm10_orientation'
        orient_name = 'speed_phi'
        self.exp.load([speed_name, orient_name])

        xy_df = self.exp.get_df('xy').loc[self.id_exp, :, :]
        xy_df.x, xy_df.y = self.exp.convert_xy_to_movie_system(self.id_exp, xy_df.x, xy_df.y)

        speed_df = self.exp.get_df(speed_name).loc[self.id_exp, :, :]/10.
        # body_orient_df = self.exp.get_df(body_orient_name).loc[self.id_exp, :, :]
        orient_df = self.exp.get_df(orient_name).loc[self.id_exp, :, :]/100.

        # diff_df = angle_distance_df(orient_df, body_orient_df).abs()
        # diff_df['temp'] = angle_distance_df(orient_df, norm_angle_tab(body_orient_df+np.pi)).abs()
        # diff_df = diff_df.min(axis=1)

        id_ants = np.array(xy_df.index.get_level_values(id_ant_name))
        y, x = np.histogram(id_ants, range(1, max(id_ants)+1))
        id_ants = list(x[:-1][y > 2*fps])

        movie = self.exp.get_movie(self.id_exp)

        body_orient_df = None
        diff_df = None
        return xy_df, speed_df, body_orient_df, orient_df, diff_df, id_ants, movie

    def get_frame_batches(self):

        self.exp.load('fps', reload=False)
        fps = self.exp.get_value('fps', self.id_exp)
        lg1 = int(self.batch_length1*fps)
        lg2 = int(self.batch_length2*fps)
        frame0 = self.xy_df0.loc[self.id_exp, self.id_ant, :].index.get_level_values(id_frame_name)[0]
        frame1 = self.xy_df0.loc[self.id_exp, self.id_ant, :].index.get_level_values(id_frame_name)[-1]
        frames = range(frame0, frame1-lg2, lg1)
        list_batch_frames = [[frame-lg1, frame+lg2] for frame in frames]

        return list_batch_frames

    def get_linearization(self):

        speed_line = self.get_sequence(self.speed_df0.loc[self.id_exp, self.id_ant, :].dropna().reset_index().values)
        # body_orient_line = self.get_sequence(
        #     self.body_orient_df0.loc[self.id_exp, self.id_ant, :].dropna().reset_index().values)
        orient_line = self.get_sequence(self.orient_df0.loc[self.id_exp, self.id_ant, :].dropna().reset_index().values)
        # diff_line = self.get_sequence(self.diff_df0.loc[self.id_exp, self.id_ant, :].dropna().reset_index().values)

        body_orient_line = None
        diff_line = None
        return speed_line, body_orient_line, orient_line, diff_line

    @staticmethod
    def get_sequence(tab, dt=5):
        tab2 = tab[:, 1].copy()
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
                res = np.delete(res, [ii, ii + 1, ii + 2])
                res = np.append(res, new_x)
                res = np.sort(res)
            else:
                res = np.delete(res, [ii, ii + 1])
            dx = res[1:] - res[:-1]
            mask = np.where(dx < dt)[0]
        res = list(set([0] + list(res) + [len(tab2) - 1]))
        res.sort()

        return tab[res, :]

    def prev_frame_batch(self):
        if self.iter_frame_batch is None:
            self.ax_ant.text(0, 0, 'No more frame batches')
            self.draw()
        else:
            self.iter_frame = 0
            self.iter_frame_batch -= 1
            if self.iter_frame_batch >= 0:
                self.frame_batch = self.list_frame_batch[self.iter_frame_batch]
            else:
                self.iter_frame_batch = len(self.list_frame_batch) - 1

            self.frame_batch = self.get_frame_batch()

            self.xy_df, self.speed_df, self.body_orient_df, self.orient_df, self.diff_df, \
                self.color_df, self.size_df, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()
            self.reset_play()

    def next_frame_batch(self):
        if self.iter_frame_batch is None:
            self.ax_ant.text(0, 0, 'No more frame batches')
            self.draw()
        else:
            self.iter_frame = 0
            self.iter_frame_batch += 1
            if self.iter_frame_batch < len(self.list_frame_batch):
                self.frame_batch = self.list_frame_batch[self.iter_frame_batch]
            else:
                self.iter_frame_batch = 0

            self.frame_batch = self.get_frame_batch()

            self.xy_df, self.speed_df, self.body_orient_df, self.orient_df, self.diff_df, \
                self.color_df, self.size_df, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()

            self.reset_play()

    def prev_ant(self):
        self.iter_id_ant -= 1
        if self.iter_id_ant < 0:
            self.iter_id_ant = len(self.list_id_ants)-1

        self.id_ant = self.get_id_ant()
        self.list_frame_batch = self.get_frame_batches()
        self.iter_frame_batch = 0
        self.frame_batch = self.get_frame_batch()
        self.speed_line, self.body_orient_line, self.orient_line, self.diff_line = self.get_linearization()

        self.xy_df, self.speed_df, self.body_orient_df, self.orient_df, self.diff_df,\
            self.color_df, self.size_df, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()
        self.reset_play()

    def next_ant(self):
        self.iter_id_ant += 1
        if self.iter_id_ant >= len(self.list_id_ants):
            self.iter_id_ant = 0

        self.id_ant = self.get_id_ant()
        self.list_frame_batch = self.get_frame_batches()
        self.iter_frame_batch = 0
        self.frame_batch = self.get_frame_batch()
        self.speed_line, self.body_orient_line, self.orient_line, self.diff_line = self.get_linearization()

        self.xy_df, self.speed_df, self.body_orient_df, self.orient_df, self.diff_df, \
            self.color_df, self.size_df, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()
        self.reset_play()

    @staticmethod
    def init_figure():
        fig = Figure()
        gs = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[2, 1, 1])
        ax = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        # ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2])
        # ax5 = fig.add_subplot(gs[3, 1])
        # fig.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=1, hspace=0.01)
        fig.subplots_adjust(left=0.08, bottom=0.1, right=0.9, top=0.9)

        ax3 = None
        ax5 = None
        return fig, ax, ax2, ax3, ax4, ax5

    def get_frame_batch(self):
        return self.list_frame_batch[self.iter_frame_batch]

    def get_id_ant(self):
        return self.list_id_ants[self.iter_id_ant]

    def time_loop(self):
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start()

    def set_canvas_size(self):
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def cropping_xy(self):

        pd_slice = pd.IndexSlice[self.id_exp, self.id_ant, self.frame_batch[0]:self.frame_batch[1]]
        xy_df = self.xy_df0.loc[pd_slice, :].copy()
        speed_df = self.speed_df0.loc[pd_slice, :].copy()
        # body_orient_df = self.body_orient_df0.loc[pd_slice, :].copy()
        orient_df = self.orient_df0.loc[pd_slice, :].copy()
        # diff_df = self.diff_df0.loc[pd_slice, :].copy()

        self.exp.load('from_outside', reload=False)

        color_df = xy_df.copy()
        color_df = color_df.drop(columns='y')
        from_outside = self.exp.get_value('from_outside', (self.id_exp, self.id_ant))
        color_df[:] = int(from_outside)
        # outside_ant_df = self.exp.from_outside.df.loc[self.id_exp, :]
        # list_outside_ant = list(outside_ant_df[outside_ant_df == 1].dropna().index)
        # color_df.loc[pd.IndexSlice[self.id_exp, list_outside_ant, :], :] = 2

        size_df = xy_df.copy()
        size_df = size_df.drop(columns='y')
        size_df[:] = 0.5
        # size_df.loc[self.id_exp, self.id_ant, :] = 5
        # color_df.loc[pd.IndexSlice[self.id_exp, self.id_ant, :], :] += 1

        if len(xy_df) > 0:
            xy = xy_df.loc[self.id_exp, self.id_ant, :]
            x0, y0 = int(np.nanmean(xy.x)), int(np.nanmean(xy.y))
        else:
            x0, y0 = 0, 0
        xy_df.x -= x0
        xy_df.y -= y0

        # xy = xy_df.loc[self.id_exp, self.id_ant, :]
        # dx = max(int(max(np.nanmax(xy.x), -np.nanmin(xy.x))), 100)
        # dy = max(int(max(np.nanmax(xy.y), -np.nanmin(xy.y))), 100)
        # dx = 500
        # dy = 500
        dx = int(self.frame_width/self.zoom)
        dy = int(self.frame_height/self.zoom)

        xy_df.x += dx
        xy_df.y += dy

        body_orient_df = None
        diff_df = None

        return xy_df, speed_df, body_orient_df, orient_df, diff_df, color_df, size_df, x0, y0, dx, dy

    def reset_play(self):
        self.iter_frame = 0

        frame = self.frame_batch[0]
        frame_img = self.crop_frame_img(self.movie.get_frame(frame))

        self.ax_ant.cla()
        self.frame_graph = self.ax_ant.imshow(frame_img, cmap='gray')

        xy = self.xy_df.loc[self.id_exp, :, :]
        color = np.array(self.color_df).ravel()
        size = np.array(self.size_df).ravel()

        self.xy_graph = self.ax_ant.scatter(
            xy.x, xy.y, c=color[:len(xy.x)], s=size[:len(xy.x)], cmap='jet', norm=colors.Normalize(0, 1))

        self.batch_text = self.ax_ant.text(
            0, 0, 'Ant: %i\nBatch: %i to %i' % (self.id_ant, self.frame_batch[0], self.frame_batch[1]),
            color='black', weight='bold', size='xx-large', horizontalalignment='center', verticalalignment='bottom')

        self.clock_text = self.ax_ant.text(
            0, 0, 'Stop', color='green', weight='bold', size='xx-large',
            horizontalalignment='right', verticalalignment='top')

        self.ax_ant.set_xlim(0, 2*self.dx)
        self.ax_ant.set_ylim(2*self.dy, 0)
        self.ax_ant.axis('off')

        self.ax_speed.cla()
        self.ax_speed.plot(self.speed_df.loc[self.id_exp, self.id_ant, :])
        self.ax_speed.grid()
        self.ax_speed.set_xlabel('Time (frame)')
        self.ax_speed.set_ylabel('v (cm/s)')

        self.ax_speed.plot(self.speed_line[:, 0], self.speed_line[:, 1], marker='o')
        self.ax_speed.set_xlim(self.frame_batch[0], self.frame_batch[1])
        self.ax_speed.set_ylim(0, 15)

        # self.ax_body_orient.cla()
        # self.ax_body_orient.plot(self.body_orient_df.loc[self.id_exp, self.id_ant, :])
        # self.ax_body_orient.grid()
        # self.ax_body_orient.set_xlabel('Time (frame)')
        # self.ax_body_orient.set_ylabel('Body orientation')
        #
        # self.ax_body_orient.plot(self.body_orient_line[:, 0], self.body_orient_line[:, 1], marker='o')
        # self.ax_body_orient.set_xlim(self.frame_batch[0], self.frame_batch[1])
        # self.ax_body_orient.set_ylim(-np.pi, np.pi)

        self.ax_orient.cla()
        self.ax_orient.plot(self.orient_df.loc[self.id_exp, self.id_ant, :])
        self.ax_orient.grid()
        self.ax_orient.set_xlabel('Time (frame)')
        self.ax_orient.set_ylabel('Orientation')

        self.ax_orient.plot(self.orient_line[:, 0], self.orient_line[:, 1], marker='o')
        self.ax_orient.set_xlim(self.frame_batch[0], self.frame_batch[1])
        self.ax_orient.set_ylim(-np.pi, np.pi)

        # self.ax_diff.cla()
        # self.ax_diff.plot(self.diff_df.loc[self.id_exp, self.id_ant, :])
        # self.ax_diff.grid()
        # self.ax_diff.set_xlabel('Time (frame)')
        # self.ax_diff.set_ylabel('Diff Orientation')
        #
        # self.ax_diff.plot(self.diff_line[:, 0], self.diff_line[:, 1], marker='o')
        # self.ax_diff.set_xlim(self.frame_batch[0], self.frame_batch[1])
        # self.ax_diff.set_ylim(0, np.pi/2.)

        self.speed_line_graph = self.ax_speed.axvline(frame, c='red')
        # self.body_orient_line_graph = self.ax_body_orient.axvline(frame, c='red')
        self.orient_line_graph = self.ax_orient.axvline(frame, c='red')
        # self.diff_line_graph = self.ax_diff.axvline(frame, c='red')

        self.draw()

    def crop_frame_img(self, frame_img):
        y0 = self.y0 - self.dy
        y1 = self.y0 + self.dy
        x0 = self.x0 - self.dx
        x1 = self.x0 + self.dx
        return frame_img[y0:y1, x0:x1]

    def update_figure(self):
        if self.play == 1:
            self.iter_frame += 1
            if self.iter_frame >= self.frame_batch[1]-self.frame_batch[0]:
                self.play = -1
            else:
                frame = self.frame_batch[0] + self.iter_frame
                frame_img = self.crop_frame_img(self.movie.get_next_frame())
                self.frame_graph.set_data(frame_img)

                pd_slice = pd.IndexSlice[self.id_exp, :, frame - 20:frame]
                xy = self.xy_df.loc[pd_slice, :]
                color = np.array(self.color_df.loc[pd_slice]).ravel()
                size = np.array(self.size_df.loc[pd_slice]).ravel()

                self.xy_graph.set_offsets(np.c_[xy.x, xy.y])
                self.xy_graph.set_array(color)
                self.xy_graph.set_sizes(size)

                self.speed_line_graph.set_xdata([frame, frame])
                # self.body_orient_line_graph.set_xdata([frame, frame])
                self.orient_line_graph.set_xdata([frame, frame])
                # self.diff_line_graph.set_xdata([frame, frame])

                self.clock_text.set_text(frame)

                self.draw()

        elif self.play == -1:
            self.reset_play()
            self.clock_text.set_text('stop')
            self.clock_text.set_color('red')
            self.play = 0
            self.draw()

    def set_to_play(self):
        self.play = 1

    def set_to_stop(self):
        self.play = 0


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, group, id_exp, outside=True):

        self.bt_height = 30
        self.bt_length = 80
        self.dl = 5

        self.exp = ExperimentGroupBuilder(root).build(group)

        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.main_widget = QtWidgets.QWidget(self)
        self.movie_canvas = MovieCanvas(self.exp, outside, self.main_widget, id_exp=id_exp)

        layout = QtWidgets.QVBoxLayout(self.main_widget)
        layout.addWidget(self.movie_canvas)

        self.setWindowTitle("Experiment"+str(self.movie_canvas.id_exp))
        self.addToolBar(QtCore.Qt.BottomToolBarArea, NavigationToolbar(self.movie_canvas, self))

        self.hbox_top = QHBoxLayout()
        self.hbox_top.addStretch(1)

        self.create_button(self.hbox_top, "Quit", self.quit)
        self.create_button(self.hbox_top, "Play", self.resume_play)
        self.create_button(self.hbox_top, "RePlay", self.replay)
        self.create_button(self.hbox_top, "Pause", self.pause_play)
        self.create_button(self.hbox_top, "Prev frames", self.prev_batch)
        self.create_button(self.hbox_top, "Next frames", self.next_batch)
        self.create_button(self.hbox_top, "Prev ant", self.prev_ant)
        self.create_button(self.hbox_top, "Next ant", self.next_ant)

        self.prev_zoom = self.movie_canvas.zoom
        self.zoom_slider = self.create_slider(self.hbox_top, 'Zoom', min_val=1, max_val=10, step=1,
                                              value=self.prev_zoom, release_func=self.set_zoom)

        layout.addLayout(self.hbox_top)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_Space:
            self.replay()
        elif key == Qt.Key_Right:
            self.next_batch()
        elif key == Qt.Key_Left:
            self.prev_batch()
        elif key == Qt.Key_Up:
            self.next_ant()
        elif key == Qt.Key_Down:
            self.prev_ant()
        else:
            pass
        self.update()

    @staticmethod
    def create_button(hbox, text, func):
        button = QPushButton(text)
        button.setFocusPolicy(QtCore.Qt.NoFocus)
        button.clicked.connect(func)
        hbox.addWidget(button)

    @staticmethod
    def create_slider(hbox, name, min_val, max_val, step, value, release_func):

        # group_box = QtWidgets.QGroupBox(name)
        slider = QSlider(Qt.Horizontal)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setTickInterval(step)
        slider.setSingleStep(step)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(value)
        slider.setFocusPolicy(QtCore.Qt.NoFocus)
        slider.sliderReleased.connect(release_func)
        # slider.sliderPressed.connect(press_func)

        label = QtWidgets.QLabel(name+':')

        # slider_box = QtWidgets.QVBoxLayout()
        # slider_box.addWidget(slider)
        #
        # group_box.setLayout(slider_box)

        hbox.addWidget(label)
        hbox.addWidget(slider)
        return slider

    def set_zoom(self):
        self.movie_canvas.zoom = self.zoom_slider
        self.movie_canvas.reset_play()

    def replay(self):
        self.movie_canvas.reset_play()
        self.movie_canvas.set_to_play()

    def pause_play(self):
        self.movie_canvas.set_to_stop()

    def resume_play(self):
        self.movie_canvas.set_to_play()

    def prev_batch(self):
        return self.movie_canvas.prev_frame_batch()

    def next_batch(self):
        self.movie_canvas.next_frame_batch()

    def prev_ant(self):
        return self.movie_canvas.prev_ant()

    def next_ant(self):
        self.movie_canvas.next_ant()

    def quit(self):
        self.close()


qApp = QtWidgets.QApplication(sys.argv)

group0 = 'UO'

aw = ApplicationWindow(group0, id_exp=40, outside=True)
aw.show()
sys.exit(qApp.exec_())
