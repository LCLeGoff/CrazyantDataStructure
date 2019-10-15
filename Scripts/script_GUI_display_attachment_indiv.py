from __future__ import unicode_literals

import sys

import numpy as np
import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QPushButton
from matplotlib import colors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from DataStructure.VariableNames import id_frame_name, id_ant_name
from ExperimentGroups import ExperimentGroups
from Scripts.root import root


class MovieCanvas(FigureCanvas):

    def __init__(self, exp: ExperimentGroups, outside=True, parent=None, id_exp=1, width=1920, height=1080*1.15):

        self.exp = exp
        self.id_exp = id_exp
        self.frame_width = width/200.
        self.frame_height = height/200.
        self.play = 0
        self.batch_length1 = 2
        self.batch_length2 = 20
        self.outside = outside

        self.exp.load(['food_radius', 'mm2px', 'fps'], reload=False)
        self.radius = self.exp.get_value('food_radius', id_exp)*self.exp.get_value('mm2px', id_exp)
        self.fps = self.exp.get_value('fps', id_exp)

        self.fig, self.ax = self.init_figure()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.xy_df0, self.food_xy_df0, self.food_angle_speed_df, self.focused_carrying_df,\
            self.not_focused_carrying_df, self.movie = self.get_traj_and_movie()

        self.list_frame_batch = self.get_frame_batches()

        self.iter_frame_batch = 0
        self.iter_frame = 0
        self.frame_graph = None
        self.xy_graph = None
        self.food_spine = None
        self.batch_text = None
        self.clock_text = None
        self.next_frame_batch()

        self.frame_batch = self.get_frame_batch()
        self.xy_df, self.food_xy_df, self.id_ant_list,\
            self.attachment_df, self.attachment_size_df, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()
        self.reset_play()

        self.set_canvas_size()
        self.time_loop()

    def get_traj_and_movie(self):
        name_outside_carrying_intervals = 'outside_ant_carrying_intervals'
        name_non_outside_carrying = 'non_outside_ant_carrying_intervals'
        name_xy = 'xy_next2food'
        self.exp.load([name_outside_carrying_intervals, name_non_outside_carrying, name_xy, 'carrying'], reload=False)

        self.exp.load_as_2d('attachment_x', 'attachment_y', 'attachment_xy', 'x', 'y', replace=True, reload=False)
        attachment_xy = self.exp.get_df('attachment_xy').loc[self.id_exp, :, :]
        carrying_df = self.exp.get_df('carrying').loc[self.id_exp, :, :]
        idx = carrying_df[carrying_df == 1].dropna().index

        xy_df = self.exp.xy_next2food.df.loc[self.id_exp, :, :]
        idx = idx.intersection(xy_df.index)
        xy_df.loc[idx] = attachment_xy.loc[idx]
        xy_df.x, xy_df.y = self.exp.convert_xy_to_movie_system(self.id_exp, xy_df.x, xy_df.y)

        outside_carrying_df = self.exp.get_df(name_outside_carrying_intervals).loc[self.id_exp, :, :]
        non_outside_carrying_df = self.exp.get_df(name_non_outside_carrying).loc[self.id_exp, :, :]

        # outside_carrying_df = outside_carrying_df[outside_carrying_df > 1].dropna()
        # non_outside_carrying_df = non_outside_carrying_df[non_outside_carrying_df > 1].dropna()

        movie = self.exp.get_movie(self.id_exp)

        self.exp.load_as_2d('food_x', 'food_y', 'food_xy', 'x', 'y', reload=False)
        food_xy_df = self.exp.food_xy.df.loc[self.id_exp, :]
        food_xy_df.x, food_xy_df.y = self.exp.convert_xy_to_movie_system(self.id_exp, food_xy_df.x, food_xy_df.y)

        self.exp.load('food_angular_speed', reload=False)
        food_angle_speed_df = self.exp.food_angular_speed.df.loc[self.id_exp, :]

        if self.outside:
            return xy_df, food_xy_df, food_angle_speed_df, outside_carrying_df, non_outside_carrying_df, movie
        else:
            return xy_df, food_xy_df, food_angle_speed_df, non_outside_carrying_df, outside_carrying_df, movie

    def get_frame_batches(self):

        self.exp.load('fps', reload=False)
        fps = self.exp.get_value('fps', self.id_exp)
        frames = self.focused_carrying_df.index.get_level_values(id_frame_name)
        list_batch_frames = []
        lg1 = int((self.batch_length1*fps) / 2)
        lg2 = int((self.batch_length2*fps) / 2)
        for frame in frames:
            list_batch_frames.append([frame - lg1, frame + lg2])
        # list_batch_frames = [[2700, 2900]]

        return list_batch_frames

    def prev_frame_batch(self):
        if self.iter_frame_batch is None:
            self.ax.text(0, 0, 'No more frame batches')
            self.draw()
        else:
            self.iter_frame = 0
            self.iter_frame_batch -= 1
            if self.iter_frame_batch >= 0:
                self.frame_batch = self.list_frame_batch[self.iter_frame_batch]
            else:
                self.iter_frame_batch = len(self.list_frame_batch) - 1

            self.frame_batch = self.get_frame_batch()

            self.xy_df, self.food_xy_df, self.id_ant_list, \
                self.attachment_df, self.attachment_size_df, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()
            self.reset_play()

    def next_frame_batch(self):
        if self.iter_frame_batch is None:
            self.ax.text(0, 0, 'No more frame batches')
            self.draw()
        else:
            self.iter_frame = 0
            self.iter_frame_batch += 1
            if self.iter_frame_batch < len(self.list_frame_batch):
                self.frame_batch = self.list_frame_batch[self.iter_frame_batch]
            else:
                self.iter_frame_batch = 0

            self.frame_batch = self.get_frame_batch()

            self.xy_df, self.food_xy_df, self.id_ant_list, \
                self.attachment_df, self.attachment_size_df, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()
            self.reset_play()

    def init_figure(self):
        fig = Figure(figsize=(self.frame_width, self.frame_height))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(0, 0.1, 1, 1)
        return fig, ax

    def get_frame_batch(self):
        frame_batch = self.list_frame_batch[self.iter_frame_batch]
        return frame_batch

    def time_loop(self):
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start()

    def set_canvas_size(self):
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def cropping_xy(self):
        food_xy_df = self.food_xy_df0.loc[self.frame_batch[0]:self.frame_batch[1]].copy()

        pd_slice = pd.IndexSlice[self.id_exp, :, self.frame_batch[0]:self.frame_batch[1]]
        xy_df = self.xy_df0.loc[pd_slice, :].copy()

        from_non_outside_color = 3*int(1-self.outside)
        from_outside_color = 3*int(self.outside)
        carrying_non_outside_color = 4*int(1-self.outside) + int(self.outside)
        carrying_outside_color = 4*int(self.outside) + int(1-self.outside)

        attachment_non_focused_color = 5*int(1-self.outside) + 2*int(self.outside)
        attachment_focused_color = 5*int(self.outside) + 2*int(1-self.outside)

        self.exp.load(['carrying', 'from_outside'], reload=False)
        carrying_df = self.exp.get_df('carrying').loc[self.id_exp, :, :]

        from_outside_df = carrying_df.copy()
        from_outside_df[:] = 0
        outside_ant_df = self.exp.from_outside.df.loc[self.id_exp, :]
        list_outside_ant = list(outside_ant_df[outside_ant_df == 1].dropna().index)
        from_outside_df.loc[pd.IndexSlice[self.id_exp, list_outside_ant, :], :] = 1

        attachment_df = carrying_df.copy()
        attachment_size_df = attachment_df.copy()
        attachment_size_df[:] = 5

        attachment_df[from_outside_df == 0] = from_non_outside_color
        attachment_df[from_outside_df == 1] = from_outside_color
        attachment_df[(carrying_df == 1)*(from_outside_df == 0)] = carrying_non_outside_color
        attachment_df[(carrying_df == 1)*(from_outside_df == 1)] = carrying_outside_color

        attachment_size_df = attachment_size_df.reindex(xy_df.index)
        attachment_df = attachment_df.reindex(xy_df.index)

        focused_attachments = self.focused_carrying_df.loc[pd_slice, :]
        for id_exp, id_ant, frame, lg in np.array(focused_attachments.reset_index()):
            attachment_df.loc[id_exp, id_ant, frame] = attachment_focused_color
            attachment_size_df.loc[id_exp, id_ant, frame] = 50

        non_focused_attachments = np.array(self.not_focused_carrying_df.loc[pd_slice, :].reset_index())
        for id_exp, id_ant, frame, lg in non_focused_attachments:
            attachment_df.loc[id_exp, id_ant, frame] = attachment_non_focused_color
            attachment_size_df.loc[id_exp, id_ant, frame] = 50

        x0, y0 = int(np.mean(xy_df.x)), int(np.mean(xy_df.y))
        print(x0, y0)
        xy_df.x -= x0
        xy_df.y -= y0
        food_xy_df.x -= x0
        food_xy_df.y -= y0
        dx = 100
        dy = 100
        xy_df.x += dx
        xy_df.y += dy
        food_xy_df.x += dx
        food_xy_df.y += dy

        list_id_ant = set(xy_df.index.get_level_values(id_ant_name))

        return xy_df, food_xy_df, list_id_ant, attachment_df, attachment_size_df, x0, y0, dx, dy

    def reset_play(self):
        self.iter_frame = 0

        frame = self.frame_batch[0]
        frame_img = self.crop_frame_img(self.movie.get_frame(frame))

        self.ax.cla()
        self.frame_graph = self.ax.imshow(frame_img, cmap='gray')

        xy = self.xy_df.loc[self.id_exp, :, :]
        food_xy = self.food_xy_df.loc[frame]
        attach = np.array(self.attachment_df).ravel()
        marker_size = np.array(self.attachment_size_df).ravel()

        self.xy_graph = self.ax.scatter(xy.x, xy.y, c=attach, s=marker_size, cmap='jet', norm=colors.Normalize(0, 5))
        self.food_spine, = self.ax.plot([food_xy.x, food_xy.x+self.radius], [food_xy.y, food_xy.y], c='r')

        self.batch_text = self.ax.text(
            self.dx, 0, 'Batch '+str(self.frame_batch[0])+' to '+str(self.frame_batch[1]),
            color='black', weight='bold', size='xx-large', horizontalalignment='center', verticalalignment='top')

        self.clock_text = self.ax.text(
            0, 0, 'Stop', color='green', weight='bold', size='xx-large',
            horizontalalignment='left', verticalalignment='top')

        self.ax.axis('equal')
        self.ax.axis('off')
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
                attach = np.array(self.attachment_df.loc[pd_slice]).ravel()
                marker_size = np.array(self.attachment_size_df.loc[pd_slice]).ravel()

                self.xy_graph.set_offsets(np.c_[xy.x, xy.y])
                self.xy_graph.set_array(attach)
                self.xy_graph.set_sizes(marker_size)

                try:
                    food_angle_speed = float(self.food_angle_speed_df.loc[frame]/self.fps)
                    x2, y2 = self.food_xy_df.loc[frame]
                    (x0, x1), (y0, y1) = self.food_spine.get_data()

                    vx = x1-x0
                    vy = y1-y0

                    speed_cos = np.cos(food_angle_speed)
                    speed_sin = np.sin(food_angle_speed)

                    spine_x = vx*speed_cos+vy*speed_sin + x2
                    spine_y = -vx*speed_sin+vy*speed_cos + y2

                    self.food_spine.set_xdata([x2, spine_x])
                    self.food_spine.set_ydata([y2, spine_y])
                except KeyError:
                    pass

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
        return self.movie_canvas.prev_frame_batch()

    def next(self):
        # self.movie_canvas.set_to_play()
        self.movie_canvas.next_frame_batch()

    def quit(self):
        self.close()


qApp = QtWidgets.QApplication(sys.argv)

group0 = 'UO'

aw = ApplicationWindow(group0, id_exp=6, outside=True)
aw.show()
sys.exit(qApp.exec_())
