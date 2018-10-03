import numpy as np
import pyqtgraph as pg
import os

from PyQt5 import QtCore, QtGui, QtWidgets
from scipy import signal, stats
from pynfb.signal_processing.filters import NotchFilter, IdentityFilter, FilterSequence

paired_colors = ['#dbae57','#57db6c','#dbd657','#57db94','#b9db57','#57dbbb','#91db57','#57d3db','#69db57','#57acdb']
images_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/../static/imag') + '/'


class SignalViewer(pg.PlotWidget):
    def __init__(self, fs, names, seconds_to_plot, overlap, signals_to_plot=None, notch_filter=False, **kwargs):
        super(SignalViewer, self).__init__(**kwargs)
        # gui settings
        self.getPlotItem().showGrid(y=True)
        self.getPlotItem().setMenuEnabled(enableMenu=False)
        self.getPlotItem().setMouseEnabled(x=False, y=False)
        self.getPlotItem().autoBtn.disable()
        self.getPlotItem().autoBtn.setScale(0)
        self.getPlotItem().setXRange(0, seconds_to_plot)
        self.setBackgroundBrush(pg.mkBrush('#252120'))

        # init buffers
        self.n_signals = len(names)
        self.n_signals_to_plot = min(self.n_signals, signals_to_plot or self.n_signals)
        self.n_samples = int(fs * seconds_to_plot) # samples to show
        self.x_stamps = np.arange(self.n_samples)
        self.previous_pos = 0 # resieved samples counter
        self.x_mesh = np.linspace(0, seconds_to_plot, self.n_samples)
        self.y_raw_buffer = np.zeros(shape=(self.n_samples, self.n_signals)) * np.nan

        # set names
        if overlap:
            self.getPlotItem().addLegend(offset=(-30, 30))

        # init signal curves
        self.curves = []
        for i in range(self.n_signals_to_plot):
            curve = pg.PlotDataItem(pen=paired_colors[i%len(paired_colors)], name=names[i])
            self.addItem(curve)
            if not overlap:
                curve.setPos(0, i + 1)
            self.curves.append(curve)

        # add vertical running line
        self.vertical_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color='B48375', width=1))
        self.addItem(self.vertical_line)

        # notch filter
        if notch_filter:
            self.notch_filter_check_box = NotchButton(self)
            self.notch_filter_check_box.setGeometry(18*2, 0, 100, 100)
            self.notch_filter = NotchFilter(50, fs, self.n_signals)
        else:
            self.notch_filter = None

    def update(self, chunk):
        # estimate current pos
        chunk_len = len(chunk)
        current_pos = (self.previous_pos + chunk_len) % self.n_samples
        current_x = self.x_mesh[current_pos]

        # notch filter
        if self.notch_filter is not None and self.notch_filter_check_box.isChecked():
            chunk = self.notch_filter.apply(chunk)

        # update buffer
        if self.previous_pos < current_pos:
            self.y_raw_buffer[self.previous_pos:current_pos] = chunk
        else:
            self.y_raw_buffer[self.previous_pos:] = chunk[:self.n_samples - self.previous_pos]
            if current_pos > 0:
                self.y_raw_buffer[:current_pos] = chunk[self.n_samples - self.previous_pos:]

        # pre-process y data and update it
        y_data = self.prepare_y_data(chunk_len)
        before_mask = (self.x_stamps < current_pos)
        for i, curve in enumerate(self.curves):
            y = y_data[:, i] if i < y_data.shape[1] else self.x_mesh * np.nan
            curve.setData(self.x_mesh, y, connect=np.isfinite(y) | before_mask)
        self.vertical_line.setValue(current_x)

        # update pos
        self.previous_pos = current_pos

    def prepare_y_data(self, chunk_len):
        return self.y_raw_buffer

    def reset_buffer(self):
        self.y_raw_buffer *= np.nan


class CuteButton(QtWidgets.QPushButton):
    """
    Black-star button
    """
    def __init__(self, parent, icon_name):
        super(CuteButton, self).__init__('', parent)
        self.setMaximumWidth(18)
        self.setMaximumHeight(18)
        self.setStyleSheet("QPushButton { background-color: #393231; color: #E5DfC5 }"
                           "QPushButton:pressed { background-color: #252120 }")
        print(images_path + icon_name)
        self.setIcon(QtGui.QIcon(images_path + icon_name))


class NotchButton(QtWidgets.QRadioButton):
    def __init__(self, parent):
        super(NotchButton, self).__init__('Notch', parent)
        self.setMaximumHeight(18)
        self.setStyleSheet("QRadioButton { background-color: #393231; color: #E5DfC5 }"
                           "QRadioButtonn:checked { background-color: #252120 }")


class RawSignalViewer(SignalViewer):
    """
    Plot raw data, each channel is on separate line
    """
    def __init__(self, fs, names, seconds_to_plot=5, **kwargs):

        super(RawSignalViewer, self).__init__(fs, names, seconds_to_plot=seconds_to_plot, overlap=False, signals_to_plot=5, **kwargs)
        # gui settings
        self.getPlotItem().setYRange(0, self.n_signals_to_plot+1)
        self.getPlotItem().disableAutoRange()

        # next previous channels groups buttons
        next_channels = CuteButton(self, 'right-arrow.png')
        next_channels.setGeometry(18, 0, 18, 25)
        prev_channels = CuteButton(self, 'left-arrow.png')
        next_channels.clicked.connect(lambda : self.next_channels_group( 1))
        prev_channels.clicked.connect(lambda : self.next_channels_group(-1))

        # attributes
        self.names = names
        self.mean = np.zeros(self.n_signals)
        self.iqr = np.ones(self.n_signals)
        self.stats_update_counter = 0
        self.indexes_to_plot = [slice(j, min(self.n_signals, j+5)) for j in range(0, self.n_signals, 5)]
        self.current_indexes_ind = 0
        self.c_slice = self.indexes_to_plot[self.current_indexes_ind]
        self.reset_labels()

    def next_channels_group(self, direction=1):
        self.y_raw_buffer *= np.nan
        self.current_indexes_ind = (self.current_indexes_ind + direction)%len(self.indexes_to_plot)
        self.c_slice = self.indexes_to_plot[self.current_indexes_ind]
        self.reset_labels()
        pass

    def reset_labels(self):
        ticks = [[(val, tick) for val, tick in zip(range(1, self.n_signals_to_plot + 1), self.names[self.c_slice])]]
        self.getPlotItem().getAxis('left').setTicks(ticks)

    def prepare_y_data(self, chunk_len):
        # update scaling stats
        self.stats_update_counter += chunk_len
        if self.stats_update_counter > self.n_samples//3:
            self.mean = np.nanmean(self.y_raw_buffer, 0)
            self.iqr = stats.iqr(self.y_raw_buffer, 0, rng=(0, 100), nan_policy='omit')
            self.iqr[self.iqr <=0 ] = 1
            self.stats_update_counter = 0

        # return scaled signals
        return ((self.y_raw_buffer - self.mean) / self.iqr)[:, self.c_slice]


class DerivedSignalViewer(SignalViewer):
    """
    Plot overlapped signals
    """
    def __init__(self, fs, names, seconds_to_plot=5, **kwargs):
        super(DerivedSignalViewer, self).__init__(fs, names, seconds_to_plot, overlap=True, **kwargs)


if __name__ == '__main__':
    fs = 250
    sec_to_plot = 10
    n_samples = sec_to_plot * fs
    n_channels = 110
    chunk_len = 8

    data = np.random.normal(size=(100000, n_channels)) * 500
    b, a = signal.butter(2, 10 / fs * 2)
    data = signal.lfilter(b, a, data, axis=0)

    a = QtWidgets.QApplication([])
    w = RawSignalViewer(fs, ['ch' + str(j) for j in range(n_channels)])

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
