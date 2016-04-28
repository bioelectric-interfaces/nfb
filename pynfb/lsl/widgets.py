# -*- coding: utf-8 -*-

import sys
import time
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
from pylsl import StreamInlet
from pylsl import resolve_stream
from scipy.fftpack import rfft, irfft, fftfreq


pg.setConfigOptions(antialias=True)
LSL_STREAM_NAMES = ['AudioCaptureWin', 'NVX136_Data', 'example']


class LSLViewer():
    def __init__(self, name='example', source_freq=500, buffer_shape=(1024, )):
        super(LSLViewer, self).__init__()
        self.source_freq = source_freq
        streams = resolve_stream('name', name)
        self.inlet = StreamInlet(streams[0], max_buflen=1, max_chunklen=8)
        self.buffer = np.zeros(buffer_shape)
        self.t0 = time.time()
        self.t = self.t0
        self.time_counter = 1
        self.w = fftfreq(buffer_shape[0], d=1. / source_freq * 2)
        self.figure = None
        self.n_channels = buffer_shape[0]

    def update_buffer(self, chunk):
        self.buffer[:-chunk.shape[0]] = self.buffer[chunk.shape[0]:]
        self.buffer[-chunk.shape[0]:] = chunk[:, :self.n_channels]

    def update(self):
        chunk, timestamp = self.inlet.pull_chunk()
        chunk = np.array(chunk)
        if chunk.shape[0] > 0:
            self.update_buffer(chunk)
            self.update_action()
        if self.time_counter % 10 == 0:
            t_curr = time.time()
            if self.figure:
                self.figure.setLabel('top', 't={:.1f}, f={:.2f}'.format(t_curr - self.t0, 1. / (t_curr - self.t) * 10))
            self.t = t_curr
        self.time_counter += 1
        pass

    def update_action(self):
        pass


class LSLPlotDataItem(pg.PlotDataItem):
    def getData(self):
        x, y = super(LSLPlotDataItem, self).getData()
        if self.opts['fftMode']:
            return x, y/(max(y) - min(y) + 1e-20)*0.75
        return x, y


class LSLRawDataWidget(LSLViewer):
    def __init__(self, n_channels=1, n_samples=500, source_freq=500, plot_freq=120):
        super(LSLRawDataWidget, self).__init__(source_freq=source_freq, buffer_shape=(n_samples, n_channels))
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.plot_freq = plot_freq
        self.init_ui()
        self.scaler = 1
        self.x_mesh = np.linspace(0, self.n_samples / self.source_freq, self.n_samples)

    def init_ui(self):
        self.figure = pg.PlotWidget()
        self.curves = []
        self.figure.setYRange(0, self.n_channels + 1)
        self.figure.setXRange(0, self.n_samples / self.source_freq)
        self.figure.showGrid(x=None, y=True, alpha=1)
        self.figure.setLabel('top', 't={:.1f}, f={:.2f}'.format(0., 0.))
        for i in range(self.n_channels):
            c = LSLPlotDataItem(pen=(i, self.n_channels * 1.3))
            self.figure.addItem(c)
            c.setPos(0, i + 1)
            self.curves.append(c)
        self.figure.show()

    def update_action(self):
        if False:
            f_signal = rfft(self.buffer, axis=0)
            cut_f_signal = f_signal.copy()
            cut_f_signal[(self.w < 7) | (self.w > 13)] = 0
            cut_signal = irfft(cut_f_signal, axis=0)
            self.buffer = cut_signal
        for i in range(self.time_counter % 1, self.n_channels, 1):
            self.curves[i].setData(self.x_mesh, self.buffer[:, i] / self.scaler)
        if self.time_counter % 10 == 0:
            self.scaler = 0.8 * self.scaler + 0.2 * (np.max(self.buffer) - np.min(self.buffer)) / 0.75


class LSLCircleFeedbackWidget(LSLViewer):
    def __init__(self, n_samples=500, n_channels=50, source_freq=500, noise_scaler=100, plot_freq=120):
        super(LSLCircleFeedbackWidget, self).__init__(buffer_shape=(n_samples, ))
        self.plot_freq = plot_freq
        self.source_freq = source_freq
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.figure = pg.plot()
        self.figure.setYRange(-5, 5)
        self.figure.hideAxis('bottom')
        self.figure.hideAxis('left')
        self.noise_scaler = noise_scaler
        self.figure.setXRange(-5, 5)
        self.x = np.linspace(-np.pi/2, np.pi/2, 100)
        self.noise = np.sin(15*self.x)*0.5-0.5
        self.noise = np.random.uniform(-0.5, 0.5, 100)-0.5
        self.p1 = self.figure.plot(np.sin(self.x),  np.cos(self.x), pen=pg.mkPen(77, 144, 254)).curve
        self.p2 = self.figure.plot(np.sin(self.x), -np.cos(self.x), pen=pg.mkPen(77, 144, 254)).curve
        fill = pg.FillBetweenItem(self.p1, self.p2, brush=(255, 255, 255, 100))
        self.figure.addItem(fill)
        self.weights = np.zeros((n_channels, ))
        self.weights[0] = 1
        self.samples_weights = np.ones((n_samples, ))

    def update_circle(self, noise_ampl):
        noise = self.noise*noise_ampl
        self.p1.setData(np.sin(self.x)*(1+noise), np.cos(self.x)*(1+noise))
        self.p2.setData(np.sin(self.x)*(1+noise), -np.cos(self.x)*(1+noise))
        pass

    def update_buffer(self, chunk):
        self.buffer[:-chunk.shape[0]] = self.buffer[chunk.shape[0]:]
        self.buffer[-chunk.shape[0]:] = np.dot(chunk[:, :self.n_channels], self.weights)

    def update_action(self):
        f_signal = rfft(self.buffer)
        cut_f_signal = f_signal.copy()
        cut_f_signal[(self.w < 8) | (self.w > 13)] = 0
        cut_f_signal = np.abs(cut_f_signal)
        noise_ampl = -np.tanh(sum(cut_f_signal) / self.noise_scaler) + 1
        self.update_circle(noise_ampl)


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.raw_view = LSLRawDataWidget(n_channels=32)
        self.fb_view = LSLCircleFeedbackWidget(n_samples=500, n_channels=1)


def main():
    app = QtGui.QApplication(sys.argv)
    win = MainWindow()
    timer = QtCore.QTimer()
    timer.timeout.connect(win.raw_view.update)
    timer.start(1000 * 1. / win.raw_view.plot_freq)
    timer1 = QtCore.QTimer()
    timer1.timeout.connect(win.fb_view.update)
    timer1.start(1000 * 1. / win.raw_view.plot_freq)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
