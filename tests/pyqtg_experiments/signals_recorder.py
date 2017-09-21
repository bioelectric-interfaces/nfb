from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
from scipy import signal



class SignalPainter(pg.PlotWidget):
    def __init__(self, fs, n_signals, seconds_to_plot=5, overlap=False, **kwargs):
        super(SignalPainter, self).__init__(**kwargs)
        # gui settings
        self.getPlotItem().hideAxis('left')
        self.getPlotItem().setMenuEnabled(enableMenu=False)
        self.getPlotItem().setMouseEnabled(x=False, y=False)

        # init signal curves
        self.curves = []
        for i in range(n_signals):
            curve = pg.PlotDataItem(pen=(2 + i % 2, 5))
            self.addItem(curve)
            if not overlap:
                curve.setPos(0, i + 1)
            self.curves.append(curve)

        # add vertical running line
        self.vertical_line = pg.PlotDataItem()
        self.addItem(self.vertical_line)

        # init buffers
        self.n_samples = fs * seconds_to_plot # samples to show
        self.n_signals = n_signals
        self.previous_pos = 0 # resieved samples counter
        self.x_mesh = np.linspace(0, seconds_to_plot, self.n_samples)
        self.y_data = np.zeros(shape=(self.n_samples, n_signals)) * np.nan

    def update(self, chunk):
        chunk_len = len(chunk)
        current_pos = (self.previous_pos + chunk_len) % self.n_samples
        current_x = self.x_mesh[current_pos]

        if self.previous_pos < current_pos:
            self.y_data[self.previous_pos:current_pos] = chunk
        else:
            self.y_data[self.previous_pos:] = chunk[:self.n_samples-self.previous_pos]
            self.y_data[:current_pos] = chunk[self.n_samples - self.previous_pos:]

        for i, curve in enumerate(self.curves):
            curve.setData(self.x_mesh, self.y_data[:, i])

        self.vertical_line.setData([current_x, current_x], [0, self.n_signals + 1])

        self.previous_pos = current_pos


if __name__ == '__main__':
    fs = 250
    sec_to_plot = 10
    n_samples = sec_to_plot * fs
    n_channels = 32
    chunk_len = 8

    data = np.random.normal(size=(100000, n_channels)) * 1
    b, a = signal.butter(2, 10 / fs * 2)
    data = signal.lfilter(b, a, data, axis=0)

    a = QtGui.QApplication([])
    w = SignalPainter(fs, n_channels)

    time = 0
    def update():
        global time
        time += chunk_len
        chunk = data[(time-chunk_len)%data.shape[0]:time%data.shape[0]]
        w.update(chunk)
    main_timer = QtCore.QTimer(a)
    main_timer.timeout.connect(update)
    main_timer.start(1000 * 1 / fs * chunk_len)

    w.showFullScreen()
    a.exec_()