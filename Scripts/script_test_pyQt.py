import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QWidget, QApplication


class KeyboardInputDemoWindow(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.code_of_last_pressed_key = 63  # The question mark ?

    def keyPressEvent(self, event):
        self.code_of_last_pressed_key = event.key()
        self.update()

    def keyReleaseEvent(self, event):
        pass
        print("key release event")

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        if self.code_of_last_pressed_key < 256:
            text_to_show_as_string = "Last pressed key: %c %X %d" % \
                                     (self.code_of_last_pressed_key,
                                      self.code_of_last_pressed_key,
                                      self.code_of_last_pressed_key)
        else:
            text_to_show_as_string = "Last pressed key: %s %X %d" % \
                                     (self.code_of_last_pressed_key,
                                      self.code_of_last_pressed_key,
                                      self.code_of_last_pressed_key)
        painter.drawText(100, 200, text_to_show_as_string)
        if self.code_of_last_pressed_key == Qt.Key_F1:
            painter.drawText(100, 250, "You pressed the F1 key")
        elif self.code_of_last_pressed_key == Qt.Key_Up:
            painter.drawText(100, 250, "You pressed the Arrow Up key")
        elif self.code_of_last_pressed_key == Qt.Key_Down:
            painter.drawText(100, 250, "You pressed the Arrow Down key")
        painter.end()


this_application = QApplication(sys.argv)
application_window = KeyboardInputDemoWindow()
application_window.show()
this_application.exec_()
