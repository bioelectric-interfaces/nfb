from PyQt4 import QtGui, QtCore
import sys
import pyqtgraph as pg
from pynfb.lsl.widgets import *
from pynfb.protocols import *

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # derived signals viewers
        roi1 = pg.PlotWidget(self)
        roi2 = pg.PlotWidget(self)
        self.roi1_curve = roi1.plot().curve
        self.roi2_curve = roi2.plot().curve
        self.roi1_buffer = np.zeros((1500,))
        # raw data viewer
        self.raw = pg.PlotWidget(self)
        self.raw_buffer = np.zeros((500, 50))
        self.scaler = 1
        self.curves = []
        self.n_channels = 50
        self.n_samples = 500
        self.source_freq = 500
        self.x_mesh = np.linspace(0, self.n_samples / self.source_freq, self.n_samples)
        self.raw.setYRange(0, 16)
        self.raw.setXRange(0, self.n_samples / self.source_freq)
        self.raw.showGrid(x=None, y=True, alpha=1)
        #self.raw.setLabel('top', 't={:.1f}, f={:.2f}'.format(0., 0.))
        for i in range(self.n_channels):
            c = LSLPlotDataItem(pen=(i, self.n_channels * 1.3))
            self.raw.addItem(c)
            c.setPos(0, i + 1)
            self.curves.append(c)
        # layout
        layout = pg.LayoutWidget(self)
        layout.addWidget(roi1, row=0, col=0)
        layout.addWidget(roi2, row=1, col=0)
        layout.addWidget(self.raw, row=2, col=0)
        layout.layout.setRowStretch(0, 1)
        layout.layout.setRowStretch(1, 1)
        layout.layout.setRowStretch(2,2)
        self.setCentralWidget(layout)
        # main window settings
        self.resize(800, 400)
        self.show()
        # subject window
        self.subject_window = SubjectWindow(self)
        self.subject_window.show()
        # time counter
        self.time_counter = 0


    def redraw_signals(self, sample, chunk):
        # derived signals
        data_buffer = self.roi1_buffer
        data_buffer[:-1] = data_buffer[1:]
        data_buffer[-1] = sample
        self.roi1_curve.setData(y=data_buffer)
        self.roi2_curve.setData(y=data_buffer[-1::-1])
        # raw signals
        self.raw_buffer[:-chunk.shape[0]] = self.raw_buffer[chunk.shape[0]:]
        self.raw_buffer[-chunk.shape[0]:] = chunk[:, :self.n_channels]
        for i in range(0, self.n_channels, 1):
            self.curves[i].setData(self.x_mesh, self.raw_buffer[:, i] / self.scaler)
        if self.time_counter % 10 == 0:
            self.scaler = 0.8 * self.scaler + 0.2 * (np.max(self.raw_buffer) - np.min(self.raw_buffer)) / 0.75
        

class SubjectWindow(QtGui.QMainWindow):
    def __init__(self, parent, **kwargs):
        super(SubjectWindow, self).__init__(parent, **kwargs)
        self.resize(500, 500)
        self.figure = pg.PlotWidget(self)
        self.setCentralWidget(self.figure)
        self.figure.setYRange(-5, 5)
        self.figure.hideAxis('bottom')
        self.figure.hideAxis('left')
        self.current_protocol = FeedbackProtocol()

    def update_protocol_state(self, sample):

        pass

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