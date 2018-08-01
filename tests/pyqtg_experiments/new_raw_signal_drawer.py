from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np
from scipy import signal

fs = 250
sec_to_plot = 10
n_samples = sec_to_plot * fs
n_channels = 32

x = np.linspace(0, sec_to_plot, n_samples)
y = np.zeros(shape=(n_channels, n_samples)) * np.nan
data = np.random.normal(size=(n_channels, 100000))*10
b, a = signal.butter(2, 10/fs*2)
data = signal.lfilter(b, a, data, axis=1)



a = QtWidgets.QApplication([])

plotWidget = pg.plot()

plotWidget.getPlotItem().hideAxis('left')
plotWidget.getPlotItem().setMenuEnabled(enableMenu=False)
plotWidget.getPlotItem().setMouseEnabled(x=False, y=False)

curves = []
for i in range(n_channels):
    c = pg.PlotDataItem(pen=(i, n_channels * 1.3), name='sf')
    plotWidget.addItem(c)
    c.setPos(0, i + 1)
    curves.append(c)

vertical_line = pg.PlotDataItem(name='sf')
plotWidget.addItem(vertical_line)

time = 0
chunk_len = 8
def update():
    global time
    global y
    time += chunk_len
    if (time-chunk_len)%n_samples< time%n_samples:
        y[:, (time-chunk_len)%n_samples:time%n_samples] = data[:, (time-chunk_len)%data.shape[1]:time%data.shape[1]]
    else:
        y[:, (time - chunk_len)%n_samples:] = data[:, (time-chunk_len)%data.shape[1]:(time-chunk_len)%data.shape[1] + n_samples - (time - chunk_len)%n_samples]
        y[:, :time%n_samples] = data[:, (time-chunk_len)%data.shape[1] + n_samples - (time - chunk_len)%n_samples:time%data.shape[1]]

    scaler = np.nanmax(y)
    for i, curve in enumerate(curves):
        curve.setData(x, y[i]/scaler)

    vertical_line.setData([x[time%n_samples], x[time%n_samples]], [0, n_channels+1])
main_timer = QtCore.QTimer(a)
main_timer.timeout.connect(update)
main_timer.start(1000*1/fs*chunk_len)

a.exec_()
