from PyQt4 import QtGui, QtCore
import sys
import pyqtgraph as pg
from pynfb.lsl.widgets import *
from pynfb.protocols import *
from pynfb.protocols_widgets import *
import time

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # timer label
        self.timer_label = QtGui.QLabel('tf')
        # derived signals viewers
        roi1 = pg.PlotWidget(self)
        roi2 = pg.PlotWidget(self)
        self.roi1_curve = roi1.plot().curve
        self.roi2_curve = roi2.plot().curve
        self.roi1_buffer = np.zeros((8000,))
        self.roi_x_mesh = np.linspace(0, 8000/ 500, 8000/8)
        # raw data viewer
        self.raw = pg.PlotWidget(self)
        self.n_channels = 50
        self.n_samples = 2000
        self.raw_buffer = np.zeros((self.n_samples, self.n_channels))
        self.scaler = 1
        self.curves = []
        self.source_freq = 500
        self.x_mesh = np.linspace(0, self.n_samples / self.source_freq, self.n_samples)
        self.raw.setYRange(0, min(8, self.n_channels))
        self.raw.setXRange(0, self.n_samples / self.source_freq)
        self.raw.showGrid(x=None, y=True, alpha=1)
        self.plot_raw_chekbox = QtGui.QCheckBox('plot raw')
        self.plot_raw_chekbox.setChecked(True)
        self.autoscale_raw_chekbox = QtGui.QCheckBox('autoscale')
        self.autoscale_raw_chekbox.setChecked(True)
        #self.raw.setLabel('top', 't={:.1f}, f={:.2f}'.format(0., 0.))
        for i in range(self.n_channels):
            c = LSLPlotDataItem(pen=(i, self.n_channels * 1.3))
            self.raw.addItem(c)
            c.setPos(0, i + 1)
            self.curves.append(c)
        # layout
        layout = pg.LayoutWidget(self)
        layout.addWidget(roi1, 0, 0, 1, 2)
        layout.addWidget(roi2, 1, 0, 1, 2)
        layout.addWidget(self.plot_raw_chekbox, 2, 0, 1, 1)
        layout.addWidget(self.autoscale_raw_chekbox, 2, 1, 1, 1)
        layout.addWidget(self.raw, 3, 0, 1, 2)
        layout.addWidget(self.timer_label, 4, 0, 1, 1)
        layout.layout.setRowStretch(0, 1)
        layout.layout.setRowStretch(1, 1)
        layout.layout.setRowStretch(3, 2)
        self.setCentralWidget(layout)
        # main window settings
        self.resize(800, 400)
        self.show()
        # subject window
        self.subject_window = SubjectWindow(self)
        self.subject_window.show()
        # time counter
        self.time_counter = 0
        self.t0 = time.time()
        self.t = self.t0

    def redraw_signals(self, sample, chunk):
        # derived signals
        data_buffer = self.roi1_buffer
        data_buffer[:-chunk.shape[0]] = data_buffer[chunk.shape[0]:]
        data_buffer[-chunk.shape[0]:] = sample
        self.roi1_curve.setData(x=self.roi_x_mesh, y=data_buffer[::8])
        self.roi2_curve.setData(x=self.roi_x_mesh, y=data_buffer[-1::-8])
        # raw signals
        if self.plot_raw_chekbox.isChecked():
            self.raw_buffer[:-chunk.shape[0]] = self.raw_buffer[chunk.shape[0]:]
            self.raw_buffer[-chunk.shape[0]:] = chunk[:, :self.n_channels]
            for i in range(0, self.n_channels, 1):
                self.curves[i].setData(self.x_mesh, self.raw_buffer[:, i] / self.scaler)
            if self.autoscale_raw_chekbox.isChecked() and self.time_counter % 10 == 0:
                self.scaler = 0.8 * self.scaler + 0.2 * (np.max(self.raw_buffer) - np.min(self.raw_buffer)) / 0.75
        # timer
        if self.time_counter % 10 == 0:
            t_curr = time.time()
            self.timer_label.setText('time:\t{:.1f}\tfps:\t{:.2f}\tchunk size:\t{}'.format(t_curr - self.t0, 1. / (t_curr - self.t) * 10, chunk.shape[0]))
            self.t = t_curr
        self.time_counter += 1

class SubjectWindow(QtGui.QMainWindow):
    def __init__(self, parent, **kwargs):
        super(SubjectWindow, self).__init__(parent, **kwargs)
        self.resize(500, 500)
        self.current_protocol = [FeedbackProtocol(), BaselineProtocol()][1]
        self.figure = self.current_protocol.widget
        self.setCentralWidget(self.figure)

    def update_protocol_state(self, sample, chunk_size=1):
        self.current_protocol.update_state(sample, chunk_size=chunk_size)
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