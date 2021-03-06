from __future__ import unicode_literals

import sys

import numpy as np
import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QPushButton, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from DataStructure.VariableNames import id_ant_name
from ExperimentGroups import ExperimentGroups
from Scripts.root import root
from Tools.PandasIndexManager.PandasIndexManager import PandasIndexManager


class MovieCanvas(FigureCanvas):

    def __init__(
            self, exp, id_exp, parent=None, training_set_name='carrying_training_set',
            width=1920, height=1080*1.15, zoom=3):

        self.training_set_name = training_set_name
        self.exp = exp
        self.zoom = zoom
        self.id_exp = id_exp
        self.frame_width = width/200.
        self.frame_height = height/200.
        self.play = 0

        self.fig, self.ax = self.init_figure()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.xy_df0, self.movie = self.get_traj_and_movie()
        self.idx_dict, self.id_ant_list = self.get_idx_list()
        self.last_treated_ant_iter = self.get_last_treated_ant()

        self.iter_ant = self.last_treated_ant_iter
        self.iter_frame = 0
        self.next_ant()

        self.frame_graph = None
        self.xy_graph = None
        self.ant_text = None
        self.clock_text = None
        self.category_text = None
        self.id_ant, self.frame_list = self.get_id_ant_and_frame_list()
        self.xy_df, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()
        self.reset_play()

        self.set_canvas_size()
        self.time_loop()

    def get_traj_and_movie(self):
        self.exp.load(self.training_set_name)
        self.exp.load('xy_next2food')
        xy_df = self.exp.xy_next2food.df.loc[self.id_exp, :, :]
        xy_df.x, xy_df.y = self.exp.convert_xy_to_movie_system(self.id_exp, xy_df.x, xy_df.y)
        movie = self.exp.get_movie(self.id_exp)
        return xy_df, movie

    def get_last_treated_ant(self):
        treated_ant_list = self.get_treated_ant_list()
        if len(treated_ant_list) == 0:
            return -1
        else:
            last_treated_ant = sorted(set(treated_ant_list))[-1]
            if last_treated_ant < len(self.id_ant_list):
                return np.where(np.array(self.id_ant_list) == last_treated_ant)[0][0]
            else:
                return -1

    def get_treated_ant_list(self):
        id_exp_ant_array = self.exp.get_array_id_exp_ant(self.training_set_name)
        id_ant_list = set(id_exp_ant_array[id_exp_ant_array[:, 0] == self.id_exp, 1])
        return list(id_ant_list)

    def get_treated_ant_category(self):
        return int(self.exp.carrying_training_set.df.loc[self.id_exp, self.id_ant, self.frame_list[0]])

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
            self.xy_df, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()
            self.reset_play()

    def next_ant(self):
        if self.iter_ant is None:
            self.ax.text(0, 0, 'No more ants')
            self.draw()
        else:
            self.iter_frame = 0
            self.iter_ant += 1
            if self.iter_ant >= len(self.id_ant_list):
                self.iter_ant = 0

            self.id_ant, self.frame_list = self.get_id_ant_and_frame_list()
            self.xy_df, self.x0, self.y0, self.dx, self.dy = self.cropping_xy()
            self.reset_play()

    def init_figure(self):
        fig = Figure(figsize=(self.frame_width, self.frame_height))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(0, 0.1, 1, 1)
        return fig, ax

    def get_idx_list(self):
        idx_dict = PandasIndexManager().get_dict_index_with_one_index_fixed(
            self.xy_df0, fixed_index_name='id_exp', fixed_index_value=self.id_exp)
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
        x0, y0 = int(np.mean(xy_df.x)), int(np.mean(xy_df.y))
        dx = int(1920 / self.zoom / 2)
        dy = int(1080 / self.zoom / 2)
        xy_df.x -= x0 - dx
        xy_df.y -= y0 - dy
        return xy_df, x0, y0, dx, dy

    def reset_play(self):
        self.iter_frame = 0

        frame = self.frame_list[self.iter_frame]
        frame_img = self.crop_frame_img(self.movie.get_frame(frame))

        self.ax.cla()
        self.frame_graph = self.ax.imshow(frame_img, cmap='gray')
        self.xy_graph, = self.ax.plot(self.xy_df.x, self.xy_df.y, '.-', lw=0.5, mew=0.5)
        self.ant_text = self.ax.text(
            self.dx, 0, 'Ant: '+str(self.id_ant), color='black', weight='bold', size='xx-large',
            horizontalalignment='center', verticalalignment='top')

        self.clock_text = self.ax.text(
            0, 0, 'Stop', color='green', weight='bold', size='xx-large',
            horizontalalignment='left', verticalalignment='top')

        if self.id_ant in self.get_treated_ant_list():
            category = str(self.get_treated_ant_category())
        else:
            category = 'None'

        self.category_text = self.ax.text(
            self.dx*2, 0, category, color='blue', weight='bold', size='xx-large',
            horizontalalignment='right', verticalalignment='top')

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
                xy = self.xy_df.loc[pd.IndexSlice[self.id_exp, self.id_ant, frame-100:frame], :]

                self.frame_graph.set_data(frame_img)
                self.xy_graph.set_xdata(xy.x)
                self.xy_graph.set_ydata(xy.y)
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


class DataManager:

    def __init__(self, exp: ExperimentGroups, id_exp, training_set_name):
        self.exp = exp
        self.id_exp = id_exp

        self.name = training_set_name
        self.exp.load('xy_next2food')
        if self.exp.is_name_in_data(self.name):
            self.exp.load(self.name)
        else:
            self.exp.add_new1d_empty(
                name=self.name, object_type='Events1d', category='BaseFood',
                label=self.name,
                description='Training set to decide if ant are carrying or not using machine learning'
            )

    def add_new_entry(self, index: pd.Index, is_carrying):
        id_exp_ant_tuples = self.exp.get_array_id_exp_ant(self.name)
        id_exp_ant_tuples = list(zip(id_exp_ant_tuples[:, 0], id_exp_ant_tuples[:, 1]))
        id_ant = index.get_level_values(id_ant_name)[0]
        if (self.exp, id_ant) in id_exp_ant_tuples:
            for id_exp, id_ant, frame in index:
                self.exp.carrying_training_set.df.loc[id_exp, id_ant, frame] = is_carrying
        else:
            df = pd.DataFrame(np.full(len(index), is_carrying, dtype=int), columns=[self.name], index=index)
            self.exp.carrying_training_set.df = self.exp.carrying_training_set.df.append(df)

    def remove_entry(self, id_ant):
        id_exp_ant_tuples = self.exp.get_array_id_exp_ant(self.name)
        id_exp_ant_tuples = list(zip(id_exp_ant_tuples[:, 0], id_exp_ant_tuples[:, 1]))
        if (self.id_exp, id_ant) in id_exp_ant_tuples:
            self.exp.carrying_training_set.df.loc[self.id_exp, id_ant, :] = None
            self.exp.carrying_training_set.df.dropna(inplace=True)

    def write_data(self):
        self.exp.write(self.name)

    def get_quantity_of_each_label(self):
        return int(self.exp.carrying_training_set.df.sum()), int((1-self.exp.carrying_training_set.df).sum())


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, group, id_exp=1, training_set_name='carrying_training_set'):

        self.bt_height = 30
        self.bt_length = 80
        self.dl = 5

        self.exp = ExperimentGroupBuilder(root).build(group)
        self.data_manager = DataManager(exp=self.exp, id_exp=id_exp, training_set_name=training_set_name)

        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.main_widget = QtWidgets.QWidget(self)
        self.movie_canvas = MovieCanvas(
            exp=self.exp, id_exp=id_exp, parent=self.main_widget, training_set_name=training_set_name)

        self.label_text = QLabel(str(self.data_manager.get_quantity_of_each_label()))

        layout = QtWidgets.QVBoxLayout(self.main_widget)
        layout.addWidget(self.movie_canvas)
        layout.addWidget(self.label_text)

        self.setWindowTitle("Experiment"+str(self.movie_canvas.id_exp))
        self.addToolBar(QtCore.Qt.BottomToolBarArea, NavigationToolbar(self.movie_canvas, self))

        self.create_button(0, 0, "Save", self.save)
        self.create_button(1, 0, "Quit", self.quit)

        self.create_button(1, 2, "Play", self.resume_play)
        self.create_button(1, 3, "RePlay", self.replay)
        self.create_button(0, 2.5, "Pause", self.pause_play)

        self.create_button(0, 5, "Prev", self.prev)
        self.create_button(0, 6, "Next", self.next)
        self.create_button(0, 8, "Carrying", self.carrying)
        self.create_button(0, 9, "Not carr.", self.not_carrying)
        self.create_button(1, 8.5, "Both", self.carrying_and_not_carrying)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_X:
            self.not_carrying()
        elif key == Qt.Key_C:
            self.carrying()
        elif key == Qt.Key_V:
            self.carrying_and_not_carrying()
        elif key == Qt.Key_Space:
            self.replay()
        elif key == Qt.Key_S:
            self.save()
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

    def carrying(self):
        self.data_manager.add_new_entry(self.movie_canvas.xy_df.index, 1)
        self.label_text.setText(str(self.data_manager.get_quantity_of_each_label()))
        self.pause_play()
        self.next()

    def not_carrying(self):
        self.data_manager.add_new_entry(self.movie_canvas.xy_df.index, 0)
        self.label_text.setText(str(self.data_manager.get_quantity_of_each_label()))
        self.pause_play()
        self.next()

    def carrying_and_not_carrying(self):
        self.data_manager.remove_entry(self.movie_canvas.id_ant)
        self.label_text.setText(str(self.data_manager.get_quantity_of_each_label()))
        self.pause_play()
        self.next()

    def prev(self):
        return self.movie_canvas.prev_ant()

    def next(self):
        # self.movie_canvas.set_to_play()
        self.movie_canvas.next_ant()

    def save(self):
        self.data_manager.write_data()
        print('save')

    def quit(self):
        self.save()
        self.close()


qApp = QtWidgets.QApplication(sys.argv)

group0 = 'UO'
aw = ApplicationWindow(group0, id_exp=2)
aw.show()
sys.exit(qApp.exec_())
