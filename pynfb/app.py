from PyQt4 import QtGui, QtCore
import sys
import pyqtgraph as pg
from pynfb.lsl.widgets import *

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.subject_window = SubjectWindow(self)
        self.subject_window.show()
        self.show()


class SubjectWindow(QtGui.QMainWindow):
    def __init__(self, parent, **kwargs):
        super(SubjectWindow, self).__init__(parent, **kwargs)
        self.resize(500, 500)
        self.figure = pg.PlotWidget(self)
        self.setCentralWidget(self.figure)
        self.figure.setYRange(-5, 5)
        self.figure.hideAxis('bottom')
        self.figure.hideAxis('left')


def main():
    app = QtGui.QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()