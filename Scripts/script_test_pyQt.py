import sys
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.qt_compat import QtWidgets
from PyQt5.QtWidgets import QSlider, QPushButton
from PyQt5 import QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):

        self.bt_height = 30
        self.bt_length = 80
        self.dl = 5

        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.main_widget = QtWidgets.QWidget(self)
        fig = Figure()
        FigureCanvas(fig)
        self.movie_canvas = FigureCanvas(fig)
        self.movie_canvas.set_parent = self.main_widget

        self.layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.layout.addWidget(self.movie_canvas)

        self.addToolBar(QtCore.Qt.BottomToolBarArea, NavigationToolbar(self.movie_canvas, self))

        self.button_layout = QtWidgets.QGridLayout()
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

        self.__add_group_box(self.button_layout)

        self.slider_layout = QtWidgets.QGridLayout()
        self.time_display_slider = self.create_slider(0, 1, 'window', min_val=40, max_val=1000, step=10, value=50)
        self.time_display_slider = self.create_slider(0, 0, 'test', min_val=40, max_val=1000, step=10, value=50)

        self.__add_group_box(self.slider_layout)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def __add_group_box(self, layout):
        group_box = QtWidgets.QGroupBox()
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def create_slider(self, n_line, n_col, name, min_val, max_val, step, value):

        groupBox = QtWidgets.QGroupBox(name)
        slider = QSlider(Qt.Horizontal)
        # slider.setFocusPolicy(Qt.StrongFocus)
        # slider.setTickPosition(QSlider.TicksBothSides)
        # slider.setTickInterval(step)
        # slider.setMinimum(min_val)
        # slider.setMaximum(max_val)
        # slider.setValue(value)

        slider_box = QtWidgets.QVBoxLayout()
        slider_box.addWidget(slider)

        groupBox.setLayout(slider_box)

        self.slider_layout.addWidget(groupBox, n_line, n_col)
        return slider

    def create_button(self, n_line, n_col, text, func):
        button = QPushButton(text)
        button.clicked.connect(func)
        self.button_layout.addWidget(button, n_line, n_col)

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

    def quit(self):
        self.close()

    def save(self):
        self.movie_canvas.write()


qApp = QtWidgets.QApplication(sys.argv)

aw = ApplicationWindow()
aw.show()
sys.exit(qApp.exec_())
