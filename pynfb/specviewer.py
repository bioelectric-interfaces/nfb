from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import numpy as np
import pyqtgraph as pg
import time

from pynfb.inlets.lsl_inlet import LSLInlet
from scipy.signal import welch

class LSLPlotDataItem(pg.PlotDataItem):
    def __init__(self, fs, *args, **kwargs):
        super(LSLPlotDataItem, self).__init__(*args, **kwargs)
        self.fs = fs

    def getData(self):
        x, y = super(LSLPlotDataItem, self).getData()
        x = self.xData
        y = self.yData
        if self.opts['fftMode']:
            print(y.shape)
            y = y[-2000:]
            x, y = welch(y, fs=500, nperseg=1000)
            # Ignore the first bin for fft data if we have a logx scale
            if self.opts['logMode'][0]:
                x = x[1:]
                y = y[1:]
        if self.opts['logMode'][0]:
            x = np.log10(x)
        if self.opts['logMode'][1]:
            y = np.log10(y)
        return x, y

class RawViewer(pg.PlotWidget):
    def __init__(self, fs, channels_labels, parent=None, buffer_time_sec=10, overlap=False):
        super(RawViewer, self).__init__(parent)
        # cross
        n_channels = len(channels_labels)
        self.fs = fs
        self.n_samples = int(buffer_time_sec * fs)
        self.n_samples_to_display = self.n_samples
        self.n_channels = n_channels
        self.raw_buffer = np.zeros((self.n_samples, n_channels)) * np.nan
        self.curves = []
        self.x_mesh = np.linspace(0, self.n_samples / fs, self.n_samples)

        self.setYRange(0, min(8, n_channels + 2))
        self.setXRange(0, self.n_samples / fs)

        self.getPlotItem().showAxis('right')
        self.getPlotItem().addLegend(offset=(-30, 30))
        for i in range(n_channels):
            c = LSLPlotDataItem(fs=self.fs, pen=(i, n_channels * 1.3), name=channels_labels[i])
            self.addItem(c)
            self.curves.append(c)


    def set_chunk(self, chunk):
        self.raw_buffer[:-chunk.shape[0]] = self.raw_buffer[chunk.shape[0]:]
        self.raw_buffer[-chunk.shape[0]:] = chunk
        self.update()

    def update(self):
        for i in range(0, self.n_channels, 1):
            self.curves[i].setData(self.x_mesh[:self.n_samples_to_display],
                                   self.raw_buffer[-self.n_samples_to_display:, i])
        #self.setXRange(0, self.x_mesh[self.n_samples_to_display-1])



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent, fs, channels_labels):
        super(MainWindow, self).__init__(parent)
        self.plt = RawViewer(fs, channels_labels)
        self.setCentralWidget(self.plt)
        self.show()



    def redraw_signals(self, chunk):
        self.plt.set_chunk(chunk)



app = QtWidgets.QApplication(sys.argv)
lsl = LSLInlet(name='NFBLab_data')
channels_labels = lsl.get_channels_labels()
fs = lsl.get_frequency()
main_timer = QtCore.QTimer(app)
w = MainWindow(None, fs, channels_labels)

def update():
    chunk = lsl.get_next_chunk()[0]
    if chunk is not None:
        w.redraw_signals(chunk)


main_timer.timeout.connect(update)
main_timer.start(1000 * 1. / fs)
sys.exit(app.exec_())
