from PyQt4 import QtGui, QtCore
import sys
import pyqtgraph as pg
from pynfb.lsl.widgets import *

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        layout = pg.LayoutWidget(self)
        roi1 = pg.PlotWidget(self)
        layout.addWidget(roi1,  row=0, col=0)
        roi2 = pg.PlotWidget(self)
        layout.addWidget(roi2,  row=1, col=0)
        raw = pg.PlotWidget(self)
        layout.addWidget(raw, row=2, col=0)
        layout.layout.setRowStretch(0, 1)
        layout.layout.setRowStretch(1, 1)
        layout.layout.setRowStretch(2,2)
        self.resize(800, 400)
        self.setCentralWidget(layout)
        self.roi1_curve = roi1.plot().curve
        self.roi2_curve = roi2.plot().curve
        self.subject_window = SubjectWindow(self)
        self.roi1_buffer = np.zeros((1500,))
        self.raw_buffer = np.zeros((1000, 30))
        self.subject_window.show()
        self.show()


    def redraw_signals(self, sample):
        data_buffer = self.roi1_buffer
        data_buffer[:-1] = data_buffer[1:]
        data_buffer[-1] = sample
        self.roi1_curve.setData(y=data_buffer)
        self.roi2_curve.setData(y=data_buffer[-1::-1])

class SubjectWindow(QtGui.QMainWindow):
    def __init__(self, parent, **kwargs):
        super(SubjectWindow, self).__init__(parent, **kwargs)
        self.resize(500, 500)
        self.figure = pg.PlotWidget(self)
        self.setCentralWidget(self.figure)
        self.figure.setYRange(-5, 5)
        self.figure.hideAxis('bottom')
        self.figure.hideAxis('left')


class Windows():
    def __init__(self):
        self.main = MainWindow()
        self.subject = self.main.subject_window


def main():
    app = QtGui.QApplication(sys.argv)
    main_window = Windows()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()