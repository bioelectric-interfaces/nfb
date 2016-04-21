from pyqtgraph.Qt import QtGui, QtCore
from pylsl import StreamInlet, resolve_stream
import numpy as np
import pyqtgraph as pg
import sys
import time

pg.setConfigOptions(antialias=False)
app = QtGui.QApplication([])
win = pg.GraphicsWindow(title="pyqtgraph example: Linked Views")
win.resize(800, 600)
win.addLabel("Linked Views", colspan=2)
n_plots = 15
n_samples = 1000
data = np.zeros((n_samples, n_plots))

curves = []
p_old = None
p = None
for j in range(n_plots):
    win.nextRow()
    p_new = win.addPlot(y=data[j])
    if j == 0: p = p_new
    if p_old:  p_new.setXLink(p_old)
    if j < n_plots - 1: p_new.showAxis('bottom', show=False)
    c = pg.PlotCurveItem(pen=(j, n_plots * 1.3))
    p_old = p_new
    p_old.addItem(c)
    curves.append(c)

streams = resolve_stream('name', 'example')  # 'AudioCaptureWin')
inlet = StreamInlet(streams[0], max_buflen=1, max_chunklen=8)
t0 = time.time()
t = t0
c = 1
chunk_counter = 0
zero_counter = 0
none_counter = 0
freq = 120


def update():
    global chunk_counter, zero_counter, none_counter, c, t, t0, data
    chunk, timestamp = inlet.pull_chunk()
    chunk = chunk
    max_samples = len(chunk)
    if max_samples>0:
        data[:-max_samples] = data[max_samples:]
        data[-max_samples:] = chunk

        for i in range(c%2, n_plots, 2):
            curves[i].setData(data[:, i])
    if chunk_counter is None:
        none_counter += 1
    if len(chunk) > 0:
        if len(chunk) != 8:
            print(len(chunk))
        chunk_counter += 1
    else:
        zero_counter += 0
    # if len(chunk) > 0:
    if c % (freq) == 0:
        t_curr = time.time()
        print('t={:.1f}, f={:.2f}, chunks={}, zeros={}, nones={}'.format(t_curr - t0,
                                                                         1. / (t_curr - t) * freq,
                                                                         chunk_counter, zero_counter, none_counter))
        t = t_curr
        chunk_counter = 0
        zero_counter = 0
        none_counter = 0
    c += 1

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1000*1./freq)

if __name__ == '__main__':
    sys.exit(app.exec_())