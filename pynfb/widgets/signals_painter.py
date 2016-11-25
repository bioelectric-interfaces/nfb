import numpy as np
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore


class CrossButtonsWidget(QtGui.QWidget):
    names = ['left', 'right', 'up', 'down']
    symbols = ['<', '>', '^', 'v']
    positions = [(1, 0), (1, 2), (0, 1), (2, 1)]

    def __init__(self, parent=None):
        super(CrossButtonsWidget, self).__init__(parent)
        buttons = dict([(name, QtGui.QPushButton(symbol)) for name, symbol in zip(self.names, self.symbols)])
        layout = QtGui.QGridLayout(self)
        self.clicked_dict = {}
        for name, pos in zip(self.names, self.positions):
            buttons[name].setAutoRepeat(True)
            buttons[name].setAutoRepeatDelay(100)
            layout.addWidget(buttons[name], *pos)
            self.clicked_dict[name] = buttons[name].clicked


class LSLPlotDataItem(pg.PlotDataItem):
    def getData(self):
        x, y = super(LSLPlotDataItem, self).getData()
        if self.opts['fftMode']:
            return x, y / (max(y) - min(y) + 1e-20) * 0.75
        return x, y

class RawViewer(pg.PlotWidget):
    def __init__(self, fs, channels_labels, parent=None, buffer_time_sec=5):
        super(RawViewer, self).__init__(parent)
        # cross
        cross = CrossButtonsWidget(self)
        cross.setGeometry(50, 0, 100, 100)
        cross.clicked_dict['up'].connect(lambda: self.update_scaler(increase=True))
        cross.clicked_dict['down'].connect(lambda: self.update_scaler(increase=False))
        cross.clicked_dict['right'].connect(lambda: self.update_n_samples_to_display(increase=True))
        cross.clicked_dict['left'].connect(lambda: self.update_n_samples_to_display(increase=False))

        n_channels = len(channels_labels)
        self.fs = fs
        self.n_samples = int(buffer_time_sec * fs)
        self.n_samples_to_display = self.n_samples
        self.n_channels = n_channels
        self.raw_buffer = np.zeros((self.n_samples, n_channels))
        self.current_pos = 0
        self.std = None
        self.scaler = 1.
        self.curves = []
        self.x_mesh = np.linspace(0, self.n_samples / fs, self.n_samples)
        self.setYRange(0, min(8, n_channels))
        self.setXRange(0, self.n_samples / fs)
        self.getPlotItem().showAxis('right')
        self.getPlotItem().getAxis('right').setTicks(
            [[(val, tick) for val, tick in zip(range(1, n_channels + 1, 2), range(1, n_channels + 1, 2))],
             [(val, tick) for val, tick in zip(range(1, n_channels + 1), range(1, n_channels + 1))]])
        self.getPlotItem().getAxis('left').setTicks(
            [[(val, tick) for val, tick in zip(range(1, n_channels + 1), channels_labels)]])
        for i in range(n_channels):
            c = LSLPlotDataItem(pen=(i, n_channels * 1.3))
            self.addItem(c)
            c.setPos(0, i + 1)
            self.curves.append(c)

    def update_std(self, chunk):
        if self.std is None:
            self.std = np.std(chunk)
        else:
            self.std = 0.5 * np.std(chunk) + 0.5 * self.std

    def set_chunk(self, chunk):
        self.raw_buffer[:-chunk.shape[0]] = self.raw_buffer[chunk.shape[0]:]
        self.raw_buffer[-chunk.shape[0]:] = chunk
        self.update()

    def update(self):
        for i in range(0, self.n_channels, 1):
            self.curves[i].setData(self.x_mesh[:self.n_samples_to_display],
                                   self.raw_buffer[-self.n_samples_to_display:, i] * self.scaler / self.std)
        self.setXRange(0, self.x_mesh[self.n_samples_to_display-1])

    def update_scaler(self, increase=False):
        step = 0.05
        self.scaler += step if increase else -step
        if self.scaler < 0:
            self.scaler = 0
        self.update()

    def update_n_samples_to_display(self, increase=False):
        step = int(self.fs * 0.5)
        self.n_samples_to_display += step if increase else -step
        if self.n_samples_to_display < 10:
            self.n_samples_to_display = 10
        elif self.n_samples_to_display > self.n_samples:
            self.n_samples_to_display = self.n_samples
        self.update()

if __name__ == '__main__':
    a = QtGui.QApplication([])
    plot_widget = RawViewer(250, ['ef', 'sf', 'qwr']*3)

    plot_widget.set_chunk(np.sin(np.arange(20)).reshape(20, 1).dot(np.ones((1, 9))))
    plot_widget.show()
    a.exec_()
