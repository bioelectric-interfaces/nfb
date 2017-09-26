from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
from scipy import signal, stats

paired_colors = ['#dbae57','#57db6c','#dbd657','#57db94','#b9db57','#57dbbb','#91db57','#57d3db','#69db57','#57acdb']


class SignalPainter(pg.PlotWidget):
    def __init__(self, fs, names, seconds_to_plot, overlap, signals_to_plot=None, **kwargs):
        super(SignalPainter, self).__init__(**kwargs)
        # gui settings
        #self.getPlotItem().hideAxis('left')
        self.getPlotItem().showGrid(y=True)
        self.getPlotItem().setMenuEnabled(enableMenu=False)
        self.getPlotItem().setMouseEnabled(x=False, y=False)
        self.getPlotItem().autoBtn.disable()
        self.getPlotItem().autoBtn.setScale(0)
        self.setBackgroundBrush(pg.mkBrush('#252120'))

        # init buffers
        self.n_signals = len(names)
        self.n_signals_to_plot = signals_to_plot or self.n_signals
        self.n_samples = int(fs * seconds_to_plot) # samples to show
        self.previous_pos = 0 # resieved samples counter
        self.x_mesh = np.linspace(0, seconds_to_plot, self.n_samples)
        self.y_raw_buffer = np.zeros(shape=(self.n_samples, self.n_signals)) * np.nan

        # set names
        if overlap:
            self.getPlotItem().addLegend(offset=(-30, 30))

        # init signal curves
        self.curves = []
        for i in range(self.n_signals_to_plot):
            curve = pg.PlotDataItem(pen=paired_colors[i%len(paired_colors)])
            self.addItem(curve)
            if not overlap:
                curve.setPos(0, i + 1)
            self.curves.append(curve)

        # add vertical running line
        self.vertical_line = pg.PlotDataItem(pen=pg.mkPen(color='B48375', width=1))
        self.vertical_line_height = [0, self.n_signals_to_plot + 1] if not overlap else [-1, 1]
        self.addItem(self.vertical_line)



    def update(self, chunk):
        chunk_len = len(chunk)
        current_pos = (self.previous_pos + chunk_len) % self.n_samples
        current_x = self.x_mesh[current_pos]

        if self.previous_pos < current_pos:
            self.y_raw_buffer[self.previous_pos:current_pos] = chunk
        else:
            self.y_raw_buffer[self.previous_pos:] = chunk[:self.n_samples - self.previous_pos]
            self.y_raw_buffer[:current_pos] = chunk[self.n_samples - self.previous_pos:]

        y_data = self.get_y_data(chunk_len)
        for i, curve in enumerate(self.curves):
            curve.setData(self.x_mesh, y_data[:, i] if i < y_data.shape[1] else self.x_mesh * np.nan)

        self.vertical_line.setData([current_x, current_x], self.vertical_line_height)

        self.previous_pos = current_pos

    def set_chunk(self, chunk):
        return self.update(chunk)

    def get_y_data(self, chunk_len):
        return self.y_raw_buffer



class CuteButton(QtGui.QPushButton):
    def __init__(self, text, parrent):
        super(CuteButton, self).__init__(text, parrent)
        self.setMaximumWidth(18)
        self.setMaximumHeight(18)
        self.setStyleSheet("QPushButton { background-color: #393231; color: #E5DfC5 }"
                          "QPushButton:pressed { background-color: #252120 }")

class RawSignalPainter(SignalPainter):
    def __init__(self, fs, names, seconds_to_plot=5, **kwargs):
        super(RawSignalPainter, self).__init__(fs, names, seconds_to_plot=seconds_to_plot, overlap=False, signals_to_plot=5, **kwargs)
        self.getPlotItem().disableAutoRange()
        self.getPlotItem().setYRange(0, self.n_signals_to_plot+1)
        self.getPlotItem().setXRange(0, seconds_to_plot)

        #self.getPlotItem().getAxis('left').setTicks(
        #    [[(val, tick) for val, tick in zip(range(1, self.n_signals + 1), names)]])
        #
        next_channels = CuteButton('->', self)
        next_channels.setGeometry(18, 0, 18, 18)
        prev_channels = CuteButton('<-', self)
        next_channels.clicked.connect(lambda : self.next_channels_group( 1))
        prev_channels.clicked.connect(lambda : self.next_channels_group(-1))

        self.names = names

        self.mean = np.zeros(self.n_signals)
        self.iqr = np.ones (self.n_signals)
        self.stats_update_counter = 0
        self.indexes_to_plot = [slice(j, min(self.n_signals, j+5)) for j in range(0, self.n_signals, 5)]
        print(self.indexes_to_plot)
        self.current_indexes_ind = 0
        self.c_slice = self.indexes_to_plot[self.current_indexes_ind]

        self.reset_labels()

    def next_channels_group(self, direction=1):
        self.y_raw_buffer *= np.nan
        self.previous_pos = 0
        self.current_indexes_ind = (self.current_indexes_ind + direction)%len(self.indexes_to_plot)
        self.c_slice = self.indexes_to_plot[self.current_indexes_ind]
        self.reset_labels()
        pass

    def reset_labels(self):
        self.getPlotItem().getAxis('left').setTicks(
            [[(val, tick) for val, tick in zip(range(1, self.n_signals_to_plot + 1),
                                               self.names[self.c_slice])]])

    def get_y_data(self, chunk_len):
        self.stats_update_counter += chunk_len
        if self.stats_update_counter > self.n_samples:
            self.mean = np.nanmean(self.y_raw_buffer, 0)
            self.iqr = stats.iqr(self.y_raw_buffer, 0, rng=(5, 95), nan_policy='omit')
            self.stats_update_counter = 0
        return ((self.y_raw_buffer - self.mean) / self.iqr)[:, self.c_slice]


if __name__ == '__main__':
    fs = 250
    sec_to_plot = 10
    n_samples = sec_to_plot * fs
    n_channels = 110
    chunk_len = 8

    data = np.random.normal(size=(100000, n_channels)) * 500
    b, a = signal.butter(2, 10 / fs * 2)
    data = signal.lfilter(b, a, data, axis=0)

    a = QtGui.QApplication([])
    w = RawSignalPainter(fs, ['ch'+str(j) for j in range(n_channels)])

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