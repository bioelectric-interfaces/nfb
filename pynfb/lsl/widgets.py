#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ZetCode PyQt4 tutorial

This example shows an icon
in the titlebar of the window.

author: Jan Bodnar
website: zetcode.com
last edited: October 2011
"""

import sys

import time
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
from pylsl import StreamInlet
from pylsl import resolve_stream


from scipy.fftpack import rfft, irfft, fftfreq




class LSLPlotDataItem(pg.PlotDataItem):

    def getData(self):
        x, y = super(LSLPlotDataItem, self).getData()
        if self.opts['fftMode']:
            return x, y/(max(y) - min(y) + 1e-20)*0.75
        return x, y


class LSLPlotWidget(pg.PlotWidget):

    def __init__(self, n_plots=1, n_samples=500, source_freq=500, plot_freq=120, n_fb_channels=5):
        super(LSLPlotWidget, self).__init__()
        self.n_plots = n_plots
        self.n_samples = n_samples
        self.source_freq = source_freq
        self.plot_freq = plot_freq
        self.x_mesh = np.linspace(0, self.n_samples / self.source_freq, self.n_samples)
        self.data = np.zeros((n_samples, n_plots))
        self.fb_buffer = np.zeros((n_samples,))
        self.weights = np.ones((n_fb_channels,)) / n_fb_channels
        self.init_ui()



    def init_ui(self):
        #self.setGeometry(300, 300, 250, 150)
        #self.setWindowTitle('LSL visualizer')
        self.curves = []
        self.setYRange(0, self.n_plots + 1)
        #self.getPlotItem().getAxis('left')
        self.setXRange(0, self.n_samples/self.source_freq)
        self.showGrid(x=None, y=True, alpha=1)
        self.setLabel('top', 't={:.1f}, f={:.2f}'.format(0., 0.))
        for i in range(self.n_plots):
            c = LSLPlotDataItem(pen=(i, self.n_plots * 1.3))
            self.addItem(c)
            c.setPos(0, i + 1)
            self.curves.append(c)
        self.plot_stream()


    def plot_stream(self):
        streams = resolve_stream('name', ['AudioCaptureWin', 'NVX136_Data', 'example'][2])  # 'AudioCaptureWin')
        self.inlet = StreamInlet(streams[0], max_buflen=1, max_chunklen=8)
        self.t0 = time.time()
        self.t = self.t0
        self.time_counter = 1
        self.scaler = 1
        self.w = fftfreq(self.n_samples, d=1./self.source_freq*2)


    def update(self):
        chunk, timestamp = self.inlet.pull_chunk()
        chunk = np.array(chunk)
        max_samples = chunk.shape[0]
        if max_samples > 0:
            self.data[:-max_samples] = self.data[max_samples:]
            self.data[-max_samples:] = chunk[:, :self.n_plots]

            if False:
                f_signal = rfft(self.data, axis=0)
                cut_f_signal = f_signal.copy()
                cut_f_signal[(self.w < 7) | (self.w > 13)] = 0
                cut_signal = irfft(cut_f_signal, axis=0)
                self.data = cut_signal

            for i in range(self.time_counter % 1, self.n_plots, 1):
                self.curves[i].setData(self.x_mesh, self.data[:, i]/self.scaler)

        if self.time_counter % 10 == 0:
            self.scaler = 0.8 * self.scaler + 0.2 * (np.max(self.data) - np.min(self.data)) / 0.75
        if self.time_counter % (10) == 0:
            self.plotItem.disableAutoRange() # TODO: move to init
            t_curr = time.time()
            #print('t={:.1f}, f={:.2f}'.format(t_curr - self.t0, 1. / (t_curr - self.t) * self.plot_freq))
            self.setLabel('top', 't={:.1f}, f={:.2f}'.format(t_curr - self.t0, 1. / (t_curr - self.t) * 10))
            self.t = t_curr
        self.time_counter += 1
        pass

class FBWidget(QtGui.QDialog):
    def __init__(self, n_samples=500, n_channels=50, source_freq=500, noise_scaler=100):
        super(FBWidget, self).__init__()
        pg.setConfigOptions(antialias=True)
        self.source_freq = source_freq
        self.fb_buffer = np.zeros((n_samples, ))
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.plt = pg.plot()
        self.plt.setYRange(-5, 5)
        self.plt.hideAxis('bottom')
        self.plt.hideAxis('left')
        self.noise_scaler = noise_scaler
        #self.getPlotItem().getAxis('left')
        self.plt.setXRange(-5, 5)
        self.x = np.linspace(-np.pi/2, np.pi/2, 100)
        self.noise = np.sin(15*self.x)*0.5#
        self.noise = np.random.uniform(-0.5, 0.5, 100)
        self.p1 = self.plt.plot(np.sin(self.x) * (1+self.noise), np.cos(self.x) * (1+self.noise), pen=pg.mkPen(None)).curve
        self.p2 = self.plt.plot(np.sin(self.x)*(1+self.noise), -np.cos(self.x)*(1+self.noise), pen=pg.mkPen(None)).curve
        #self.p1.curve.getPath()
        #self.p2.curve.getPath()
        fill = pg.FillBetweenItem(self.p1, self.p2, brush=(77, 144, 254, 255))
        self.plt.addItem(fill)
        #self.plot.show()
        self.weights = np.zeros((n_channels, ))
        self.weights[0] = 1
        self.plot_stream()

    def plot_stream(self):
        streams = resolve_stream('name', ['AudioCaptureWin', 'NVX136_Data', 'example'][2])  # 'AudioCaptureWin')
        self.inlet = StreamInlet(streams[0], max_buflen=1, max_chunklen=8)
        self.t0 = time.time()
        self.t = self.t0
        self.time_counter = 1
        self.scaler = 1
        self.w = fftfreq(self.n_samples, d=1./self.source_freq*2)

    def update_circle(self, noise_ampl):
        noise = self.noise*noise_ampl
        self.p1.setData(np.sin(self.x)*(1+noise), np.cos(self.x)*(1+noise))
        self.p2.setData(np.sin(self.x) *(1+noise), -np.cos(self.x) * (1+noise))
        pass

    def update(self):
        chunk, timestamp = self.inlet.pull_chunk()
        chunk = np.array(chunk)
        max_samples = chunk.shape[0]
        if max_samples > 0:
            self.fb_buffer[:-max_samples] = self.fb_buffer[max_samples:]
            self.fb_buffer[-max_samples:] = np.dot(chunk[:, :self.n_channels], self.weights)
            f_signal = rfft(self.fb_buffer)
            cut_f_signal = f_signal.copy()
            cut_f_signal[(self.w < 8) | (self.w > 13)] = 0
            cut_f_signal = np.abs(irfft(cut_f_signal))
            noise_ampl = -np.tanh(sum(cut_f_signal)/self.noise_scaler)+1
            self.update_circle(noise_ampl)
        if self.time_counter % (10) == 0:
            t_curr = time.time()
            self.plt.setLabel('top', 't={:.1f}, f={:.2f}'.format(t_curr - self.t0, 1. / (t_curr - self.t) * 10))
            self.t = t_curr
        self.time_counter += 1
        pass


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.raw_view = LSLPlotWidget(n_plots=50)
        self.fb_view = FBWidget(n_samples=500, n_channels=1)
        self.raw_view.show()


def main():

    app = QtGui.QApplication(sys.argv)
    #widget = LSLPlotWidget(n_plots=50)
    #fb = FBWidget()
    #widget.show()
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