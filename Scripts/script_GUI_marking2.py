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

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from DataStructure.VariableNames import id_frame_name, id_ant_name
from ExperimentGroups import ExperimentGroups
from Scripts.root import root


class MovieCanvas(FigureCanvas):

    def __init__(self, exp: ExperimentGroups, outside=True, parent=None, id_exp=1):

        self.exp = exp
        self.id_exp = id_exp
        self.frame_width = 1920
        self.frame_height = 1080
        self.zoom = 10
        self.play = 0
        self.time_interval = 0.5
        self.outside = outside

        self.plot_markings = True
        self.plot_traj = True

        self.fig, self.ax_ant = self.init_figure()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.xy_df0, self.list_id_ants, self.movie, self.fps, self.markings = self.get_traj_and_movie()

        self.iter_id_ant = 0
        self.id_ant = self.get_id_ant()
        self.list_marking_frames = self.get_marking_frames()

        self.frame_graph = None
        self.xy_graph = None
        self.marking_graph = None

        self.batch_text = None
        self.clock_text = None

        self.iter_frame = 0
        self.iter_marking_frame = -1

        self.next_marking_frame()

        self.xy_df, self.color_df, self.size_df = None, None, None
        self.x0, self.y0, self.dx, self.dy = None, None, None, None
        self.marking_frame, self.marking_frame0, self.marking_frame1 = None, None, None
        self.get_marking_frame()
        self.refresh()

        self.set_canvas_size()
        self.time_loop()

    def get_traj_and_movie(self):
        movie = self.exp.get_movie(self.id_exp)

        self.exp.load_as_2d('mm10_x', 'mm10_y', 'xy', 'x', 'y', replace=True, reload=False)
        xy_df = self.exp.get_df('xy').loc[self.id_exp, :, :]
        xy_df.x, xy_df.y = self.exp.convert_xy_to_movie_system(self.id_exp, xy_df.x, xy_df.y)

        self.exp.load(['potential_marking_intervals', 'fps'])
        fps = self.exp.get_value('fps', self.id_exp)

        markings = self.exp.get_df('potential_marking_intervals').loc[self.id_exp, :, :]

        id_ants = list(set(markings.index.get_level_values(id_ant_name)))
        id_ants.sort()
        return xy_df, id_ants, movie, fps, markings

    def get_marking_frames(self):
        markings_frames = list(self.markings.loc[self.id_exp, self.id_ant, :].index.get_level_values(id_frame_name))
        return markings_frames

    def prev_marking_frame(self):
        self.iter_frame = 0

        self.iter_marking_frame -= 1
        if self.iter_marking_frame < 0:
            self.iter_marking_frame = len(self.list_marking_frames) - 1

        self.get_marking_frame()

        self.refresh()

    def refresh(self):
        self.xy_df, self.color_df, self.size_df, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()
        self.reset_play()

    def next_marking_frame(self):
        self.iter_frame = 0

        self.iter_marking_frame += 1
        if self.iter_marking_frame >= len(self.list_marking_frames):
            self.iter_marking_frame = 0

        self.get_marking_frame()

        self.refresh()

    def prev_ant(self):
        self.iter_id_ant -= 1
        if self.iter_id_ant < 0:
            self.iter_id_ant = len(self.list_id_ants)-1

        self.id_ant = self.get_id_ant()
        self.list_marking_frames = self.get_marking_frames()
        self.iter_marking_frame = 0
        self.get_marking_frame()

        self.refresh()

    def next_ant(self):
        self.iter_id_ant += 1
        if self.iter_id_ant >= len(self.list_id_ants):
            self.iter_id_ant = 0

        self.id_ant = self.get_id_ant()
        self.list_marking_frames = self.get_marking_frames()
        self.iter_marking_frame = 0
        self.get_marking_frame()

        self.refresh()

    @staticmethod
    def init_figure():
        fig = Figure()
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        ax = fig.add_subplot(gs[:, 0])
        return fig, ax

    def get_marking_frame(self):
        frame = self.list_marking_frames[self.iter_marking_frame]
        lg = int(self.fps*self.time_interval)
        self.marking_frame, self.marking_frame0, self.marking_frame1 = frame, frame-lg, frame+lg

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

        pd_slice = pd.IndexSlice[self.id_exp, self.id_ant, self.marking_frame0:self.marking_frame1]
        xy_df = self.xy_df0.loc[pd_slice, :].copy()

        self.exp.load('from_outside', reload=False)

        color_df = xy_df.copy()
        color_df = color_df.drop(columns='y')
        color_df[:] = np.nan
        color_df.loc[self.id_exp, self.id_ant, self.list_marking_frames] = 0
        color_df.loc[self.id_exp, self.id_ant, self.marking_frame] = 1

        size_df = xy_df.copy()
        size_df = size_df.drop(columns='y')
        size_df[:] = np.nan
        size_df.loc[self.id_exp, self.id_ant, self.list_marking_frames] = 10
        size_df.loc[self.id_exp, self.id_ant, self.marking_frame] = 20

        if len(xy_df) > 0:
            xy = xy_df.loc[self.id_exp, self.id_ant, :]
            x0, y0 = int(np.nanmean(xy.x)), int(np.nanmean(xy.y))
        else:
            x0, y0 = 0, 0
        xy_df.x -= x0
        xy_df.y -= y0

        dx = int(self.frame_width/self.zoom)
        dy = int(self.frame_height/self.zoom)

        xy_df.x += dx
        xy_df.y += dy

        return xy_df, color_df, size_df, x0, y0, dx, dy

    def reset_play(self):
        self.iter_frame = 0

        frame_img = self.crop_frame_img(self.movie.get_frame(self.marking_frame0))

        self.ax_ant.cla()
        self.frame_graph = self.ax_ant.imshow(frame_img, cmap='gray')

        xy = self.xy_df.loc[self.id_exp, :, :]
        color = np.array(self.color_df).ravel()
        size = np.array(self.size_df).ravel()

        norm = colors.Normalize(0, 1)
        self.xy_graph = self.ax_ant.scatter(xy.x, xy.y, c='k', s=0.5)
        self.marking_graph = self.ax_ant.scatter(
            xy.x, xy.y, c=color[:len(xy.x)], s=size[:len(xy.x)], cmap='bwr', norm=norm)

        from_outside = self.exp.get_value('from_outside', (self.id_exp, self.id_ant))
        if from_outside == 1:
            ant_type = 'outside'
        else:
            ant_type = 'inside'
        self.batch_text = self.ax_ant.text(
            0, 0, '%s ant: %i\nMarking: %i' % (ant_type, self.id_ant, self.marking_frame),
            color='black', weight='bold', size='xx-large', horizontalalignment='center', verticalalignment='bottom')

        self.clock_text = self.ax_ant.text(
            0, 0, 'Stop', color='green', weight='bold', size='xx-large',
            horizontalalignment='right', verticalalignment='top')

        self.ax_ant.set_xlim(0, 2*self.dx)
        self.ax_ant.set_ylim(2*self.dy, 0)
        self.ax_ant.axis('off')

        self.draw()

    def crop_frame_img(self, frame_img):
        y0 = max(self.y0 - self.dy, 0)
        y1 = min(self.y0 + self.dy, 1079)
        x0 = max(self.x0 - self.dx, 0)
        x1 = min(self.x0 + self.dx, 1919)
        return frame_img[y0:y1, x0:x1]

    def update_figure(self):
        if self.play == 1:
            self.iter_frame += 1
            if self.iter_frame >= self.marking_frame1-self.marking_frame0:
                self.play = -1
            else:
                frame = self.marking_frame0 + self.iter_frame
                frame_img = self.crop_frame_img(self.movie.get_next_frame())
                self.frame_graph.set_data(frame_img)

                pd_slice = pd.IndexSlice[self.id_exp, :, frame - 20:frame]
                xy = self.xy_df.loc[pd_slice, :]
                color = np.array(self.color_df.loc[pd_slice]).ravel()
                size = np.array(self.size_df.loc[pd_slice]).ravel()

                if self.plot_traj is True:
                    self.xy_graph.set_offsets(np.c_[xy.x, xy.y])
                else:
                    self.xy_graph.set_offsets(np.zeros((0, 2)))

                if self.plot_markings is True:
                    self.marking_graph.set_offsets(np.c_[xy.x, xy.y])
                    self.marking_graph.set_array(color)
                    self.marking_graph.set_sizes(size)
                else:
                    self.marking_graph.set_offsets(np.zeros((0, 2)))

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

        layout.addLayout(self.hbox_top)

        self.hbox_bottom = QHBoxLayout()
        self.hbox_bottom.addStretch(1)

        self.zoom_slider = self.create_slider(self.hbox_bottom, 'Zoom', min_val=1, max_val=20, step=1,
                                              value=self.movie_canvas.zoom, func=self.set_zoom)

        self.time_interval_slider = self.create_slider(self.hbox_bottom, 'Time', min_val=10, max_val=100, step=10,
                                                       value=self.movie_canvas.time_interval*100,
                                                       func=self.set_time_interval)
        self.traj_checkbox = self.create_checkbox(
            self.hbox_bottom, "Plot traj", self.set_plot_traj, self.movie_canvas.plot_traj)
        self.marking_checkbox = self.create_checkbox(
            self.hbox_bottom, "Plot markings", self.set_plot_markings, self.movie_canvas.plot_markings)
        layout.addLayout(self.hbox_bottom)

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
    def create_checkbox(hbox, text, func, val):
        checkbox = QtWidgets.QCheckBox(text)
        checkbox.setChecked(val)
        checkbox.setFocusPolicy(QtCore.Qt.NoFocus)
        checkbox.stateChanged.connect(func)
        hbox.addWidget(checkbox)
        return checkbox

    @staticmethod
    def create_slider(hbox, name, min_val, max_val, step, value, func):

        slider = QSlider(Qt.Horizontal)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setTickInterval(step)
        slider.setSingleStep(step)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(value)
        slider.setFocusPolicy(QtCore.Qt.NoFocus)
        slider.valueChanged.connect(func)
        slider.sliderReleased.connect(func)

        label = QtWidgets.QLabel(name+':')

        hbox.addWidget(label)
        hbox.addWidget(slider)
        return slider

    def set_zoom(self):
        self.movie_canvas.zoom = self.zoom_slider.value()
        self.movie_canvas.refresh()

    def set_time_interval(self):
        self.movie_canvas.time_interval = self.time_interval_slider.value()/100.
        self.movie_canvas.get_marking_frame()
        self.movie_canvas.refresh()

    def set_plot_traj(self):
        self.movie_canvas.plot_traj = self.traj_checkbox.isChecked()
        self.movie_canvas.refresh()

    def set_plot_markings(self):
        self.movie_canvas.plot_markings = self.marking_checkbox.isChecked()
        self.movie_canvas.refresh()

    def replay(self):
        self.movie_canvas.reset_play()
        self.movie_canvas.set_to_play()
        self.movie_canvas.refresh()

    def pause_play(self):
        self.movie_canvas.set_to_stop()

    def resume_play(self):
        self.movie_canvas.set_to_play()

    def prev_batch(self):
        return self.movie_canvas.prev_marking_frame()

    def next_batch(self):
        self.movie_canvas.next_marking_frame()

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
