from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
from scipy import signal

paired_colors = ['#dbae57','#57db6c','#dbd657','#57db94','#b9db57','#57dbbb','#91db57','#57d3db','#69db57','#57acdb']


class SignalPainter(pg.PlotWidget):
    def __init__(self, fs, names, seconds_to_plot=5, overlap=False, **kwargs):
        super(SignalPainter, self).__init__(**kwargs)
        # gui settings
        #self.getPlotItem().hideAxis('left')
        self.getPlotItem().showGrid(y=True)
        self.getPlotItem().setMenuEnabled(enableMenu=False)
        self.getPlotItem().setMouseEnabled(x=False, y=False)
        self.setBackgroundBrush(pg.mkBrush('#252120'))

        # init buffers
        self.n_signals = len(names)
        self.n_samples = int(fs * seconds_to_plot) # samples to show
        self.previous_pos = 0 # resieved samples counter
        self.x_mesh = np.linspace(0, seconds_to_plot, self.n_samples)
        self.y_raw_buffer = np.zeros(shape=(self.n_samples, self.n_signals)) * np.nan

        # set names
        if not overlap:
            self.getPlotItem().getAxis('left').setTicks(
                [[(val, tick) for val, tick in zip(range(1, self.n_signals + 1), names)]])
        else:
            self.getPlotItem().addLegend(offset=(-30, 30))

        # init signal curves
        self.curves = []
        for i in range(self.n_signals):
            print(names[i] if names else '')
            curve = pg.PlotDataItem(pen=paired_colors[i%len(paired_colors)], name=names[i] if names else '')
            self.addItem(curve)
            if not overlap:
                curve.setPos(0, i + 1)
            self.curves.append(curve)

        # add vertical running line
        self.vertical_line = pg.PlotDataItem(pen=pg.mkPen(color='B48375', width=1))
        self.vertical_line_height = [0, self.n_signals + 1] if not overlap else [-1, 1]
        self.addItem(self.vertical_line)



    def update(self, chunk):
        chunk_len = len(chunk)
        current_pos = (self.previous_pos + chunk_len) % self.n_samples
        current_x = self.x_mesh[current_pos]
        chunk = self.modify(chunk)

        if self.previous_pos < current_pos:
            self.y_raw_buffer[self.previous_pos:current_pos] = chunk
        else:
            self.y_raw_buffer[self.previous_pos:] = chunk[:self.n_samples - self.previous_pos]
            self.y_raw_buffer[:current_pos] = chunk[self.n_samples - self.previous_pos:]

        for i, curve in enumerate(self.curves):
            curve.setData(self.x_mesh, self.y_raw_buffer[:, i])

        self.vertical_line.setData([current_x, current_x], self.vertical_line_height)

        self.previous_pos = current_pos

    def modify(self, chunk):
        return chunk

    def update_std(self, chunk):
        pass

    def set_chunk(self, chunk):
        self.update(chunk)

    def update_levels(self):
        pass

if __name__ == '__main__':
    fs = 250
    sec_to_plot = 10
    n_samples = sec_to_plot * fs
    n_channels = 10
    chunk_len = 8

    data = np.random.normal(size=(100000, n_channels)) * 1
    b, a = signal.butter(2, 10 / fs * 2)
    data = signal.lfilter(b, a, data, axis=0)

    a = QtGui.QApplication([])
    w = SignalPainter(fs, ['ch'+str(j) for j in range(n_channels)])

    time = 0
    def update():
        global time
        time += chunk_len
        chunk = data[(time-chunk_len)%data.shape[0]:time%data.shape[0]]
        w.update(chunk)
    main_timer = QtCore.QTimer(a)
    main_timer.timeout.connect(update)
    main_timer.start(1000 * 1 / fs * chunk_len)

    w.show()
    a.exec_()