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
                cut_f_signal[(self.w > 45) & (self.w < 60)] = 0
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
    def __init__(self, n_samples=500, n_channels=3):
        super(FBWidget, self).__init__()
        self.fb_buffer = np.zeros((n_samples, ))
        self.plot = LSLPlotWidget(n_plots=1)
        self.plot.show()
        self.weights = np.ones((n_channels, ))/n_channels

    def update_fb(self):
        y = np.dot(self.plot.data, self.weights)
        pass

class App(QtGui.QMainWindow):
    def __init__(self):
        super(App, self).__init__()
        self.widget = LSLPlotWidget(n_plots=50)
        self.fb = FBWidget(n_channels=50)
        self.widget.show()

    def update_fb(self):
        y = np.dot(self.widget.data, self.fb.weights)
        f_signal = rfft(y)
        cut_f_signal = f_signal.copy()
        cut_f_signal[(self.fb.plot.w < 7) | (self.fb.plot.w > 13)] = 0
        #print(cut_f_signal.shape)
        cut_signal = np.abs(irfft(cut_f_signal))
        self.fb.plot.curves[0].setData(self.fb.plot.x_mesh, np.abs(cut_f_signal))


def main():

    app = QtGui.QApplication(sys.argv)
    #widget = LSLPlotWidget(n_plots=50)
    #fb = FBWidget()
    #widget.show()
    win = App()
    timer = QtCore.QTimer()
    timer.timeout.connect(win.widget.update)
    timer.start(1000 * 1. / win.widget.plot_freq)

    timer1 = QtCore.QTimer()
    timer1.timeout.connect(win.update_fb)
    timer1.start(1000 * 1. / 100)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()