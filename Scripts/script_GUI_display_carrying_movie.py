from __future__ import unicode_literals

import sys

import numpy as np
import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Scripts.root import root
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class MovieCanvas(FigureCanvas):

    def __init__(self, exp, parent=None, id_exp=1, width=1920, height=1080*1.15, zoom=4):

        self.exp = exp
        self.zoom = zoom
        self.id_exp = id_exp
        self.frame_width = width/200.
        self.frame_height = height/200.
        self.play = 0

        self.fig, self.ax = self.init_figure()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.xy_df0, self.carrying_df0, self.movie = self.get_traj_and_movie(id_exp)
        self.idx_dict, self.id_ant_list = self.get_idx_list(id_exp)

        self.iter_ant = 0
        self.iter_frame = 0
        self.next_ant()

        self.frame_graph = None
        self.xy_graph = None
        self.ant_text = None
        self.clock_text = None
        self.id_ant, self.frame_list = self.get_id_ant_and_frame_list()
        self.xy_df, self.carrying_df, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()
        self.reset_play()

        self.set_canvas_size()
        self.time_loop()

    def get_traj_and_movie(self, id_exp):
        name_carrying = 'carrying'
        self.exp.load(name_carrying)
        self.exp.load('xy_next2food')
        xy_df = self.exp.xy_next2food.df.loc[id_exp, :, :]
        xy_df.x, xy_df.y = self.exp.convert_xy_to_movie_system(self.id_exp, xy_df.x, xy_df.y)
        carrying_df = self.exp.get_df(name_carrying)
        movie = self.exp.get_movie(id_exp)
        return xy_df, carrying_df, movie

    def prev_ant(self):
        if self.iter_ant is None:
            self.ax.text(0, 0, 'No more ants')
            self.draw()
        else:
            self.iter_frame = 0
            self.iter_ant -= 1
            if self.iter_ant >= 0:
                self.id_ant = self.id_ant_list[self.iter_ant]
            else:
                self.iter_ant = len(self.id_ant_list)-1

            self.id_ant, self.frame_list = self.get_id_ant_and_frame_list()
            self.xy_df, self.carrying_df, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()
            self.reset_play()

    def next_ant(self):
        if self.iter_ant is None:
            self.ax.text(0, 0, 'No more ants')
            self.draw()
        else:
            self.iter_frame = 0
            self.iter_ant += 1
            if self.iter_ant < len(self.id_ant_list):
                self.id_ant = self.id_ant_list[self.iter_ant]
            else:
                self.iter_ant = 0

            self.id_ant, self.frame_list = self.get_id_ant_and_frame_list()
            self.xy_df, self.carrying_df, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()
            self.reset_play()

    def init_figure(self):
        fig = Figure(figsize=(self.frame_width, self.frame_height))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(0, 0.1, 1, 1)
        return fig, ax

    def get_idx_list(self, id_exp):
        idx_dict = PandasIndexManager().get_dict_index_with_one_index_fixed(self.xy_df0, 'id_exp', id_exp)
        list_id_ant = sorted(idx_dict.keys())
        return idx_dict, list_id_ant

    def get_id_ant_and_frame_list(self):
        id_ant = self.id_ant_list[self.iter_ant]
        frame_list = self.idx_dict[id_ant]
        return id_ant, frame_list

    def time_loop(self):
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start()

    def set_canvas_size(self):
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def cropping_xy(self):
        xy_df = self.xy_df0.loc[pd.IndexSlice[self.id_exp, self.id_ant, :], :]
        carrying_df = self.carrying_df0.loc[pd.IndexSlice[self.id_exp, self.id_ant, :], :]
        xy_df = xy_df.reindex(carrying_df.index)
        x0, y0 = int(np.mean(xy_df.x)), int(np.mean(xy_df.y))
        dx = int(1920 / self.zoom / 2)
        dy = int(1080 / self.zoom / 2)
        xy_df.x -= x0 - dx
        xy_df.y -= y0 - dy
        return xy_df, carrying_df, x0, y0, dx, dy

    def reset_play(self):
        self.iter_frame = 0

        frame = self.frame_list[self.iter_frame]
        frame_img = self.crop_frame_img(self.movie.get_frame(frame))
        xy = self.xy_df.loc[pd.IndexSlice[self.id_exp, self.id_ant, :], :]
        carr = np.array(self.carrying_df.loc[pd.IndexSlice[self.id_exp, self.id_ant, :]])[:, 0]

        self.ax.cla()
        self.frame_graph = self.ax.imshow(frame_img, cmap='gray')
        self.xy_graph = self.ax.scatter(xy.x, xy.y, c=carr, alpha=0.2)
        self.ant_text = self.ax.text(
            self.dx, 0, 'Ant: '+str(self.id_ant), color='black', weight='bold', size='xx-large',
            horizontalalignment='center', verticalalignment='top')

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
            self.iter_frame += 1
            if self.iter_frame >= len(self.frame_list):
                self.play = -1
            else:
                frame = self.frame_list[self.iter_frame]

                frame_img = self.crop_frame_img(self.movie.get_next_frame())
                for i in range(self.frame_list[self.iter_frame - 1] + 1, frame):
                    frame_img = self.crop_frame_img(self.movie.get_next_frame())
                xy = self.xy_df.loc[pd.IndexSlice[self.id_exp, self.id_ant, frame-20:frame], :]
                carr = np.array(
                    self.carrying_df.loc[
                        pd.IndexSlice[pd.IndexSlice[self.id_exp, self.id_ant, frame-20:frame]]])[:, 0]

                self.frame_graph.set_data(frame_img)
                self.xy_graph.set_offsets(np.c_[xy.x, xy.y])
                self.xy_graph.set_array(carr)
                self.clock_text.set_text(frame)

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

        self.create_button(1, 0, "Quit", self.quit)

        self.create_button(1, 2, "Play", self.resume_play)
        self.create_button(1, 3, "RePlay", self.replay)
        self.create_button(0, 2.5, "Pause", self.pause_play)

        self.create_button(0, 5, "Prev", self.prev)
        self.create_button(0, 6, "Next", self.next)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_Space:
            self.replay()
        elif key == Qt.Key_Right:
            self.next()
        elif key == Qt.Key_Left:
            self.prev()
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

    def prev(self):
        return self.movie_canvas.prev_ant()

    def next(self):
        # self.movie_canvas.set_to_play()
        self.movie_canvas.next_ant()

    def quit(self):
        self.close()


qApp = QtWidgets.QApplication(sys.argv)

group0 = 'UO'

aw = ApplicationWindow(group0, id_exp=1)
aw.show()
sys.exit(qApp.exec_())
