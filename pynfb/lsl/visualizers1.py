# -*- coding: utf-8 -*-

from pyqtgraph.Qt import QtGui, QtCore
from pylsl import StreamInlet, resolve_stream
import numpy as np
import pyqtgraph as pg
from pyqtgraph.ptime import time
pg.setConfigOptions(antialias=False)
app = QtGui.QApplication([])


win = pg.GraphicsWindow(title="pyqtgraph example: Linked Views")
win.resize(800,600)

win.addLabel("Linked Views", colspan=2)
nPlots = 2
nSamples = 500*2
data1 = np.zeros((nPlots, nSamples))

curves = []
p_old = None
p = None
for j in range(nPlots):
    win.nextRow()
    p_new = win.addPlot(y=data1[j])
    if j==0:
        p= p_new
    if p_old:
        p_new.setXLink(p_old)
    if j < nPlots-1:
        p_new.showAxis('bottom', show=False)
    c = pg.PlotCurveItem(pen=(j, nPlots * 1.3))
    p_old = p_new
    p_old.addItem(c)
    curves.append(c)


data = np.random.normal(size=(nPlots*23, nSamples))

ptr = 0
lastTime = time()
fps = None
count = 0



print("looking for an EEG stream...")
streams = resolve_stream('name', 'example') #'AudioCaptureWin')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0], max_buflen=1)


def update():
    global curve, data, ptr, p_old, lastTime, fps, nPlots, count
    count += 1
    #print "---------", count
    #sample, timestamp = inlet.pull_sample(0.0)
    max_samples = 1
    if count%1==0:
        chunk, timestamp1 = inlet.pull_chunk(0.0, max_samples=max_samples)
        #print(len(chunk), len(chunk[0]))
    data1[:, :-max_samples] = data1[:, max_samples:]
    data1[:, -max_samples:] = np.array(chunk).T if chunk else np.nan*np.ones(nPlots)

    for i in range(nPlots):
        #curves[i].setData(data[(ptr+i)%data.shape[0]])
        curves[i].setData(data1[i])

    #print "   setData done."
    ptr += nPlots
    now = time()
    dt = now - lastTime
    lastTime = now
    if fps is None:
        fps = 1.0/dt
    else:
        s = np.clip(dt*3., 0, 1)
        fps = fps * (1-s) + (1.0/dt) * s
    p.setTitle('%0.2f fps' % fps)
    #app.processEvents()  ## force complete redraw for every plot
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1000*1./10000)



## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
