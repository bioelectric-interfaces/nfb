# -*- coding: utf-8 -*-

from pyqtgraph.Qt import QtGui, QtCore
from pylsl import StreamInlet, resolve_stream
import numpy as np
import pyqtgraph as pg
from pyqtgraph.ptime import time
pg.setConfigOptions(antialias=False)
app = QtGui.QApplication([])





p_old = pg.plot()
p_old.setWindowTitle('pyqtgraph example: MultiPlotSpeedTest')
p_old.setLabel('bottom', 'Index', units='B')

nPlots = 8
nSamples = 500
curves = []
for i in range(nPlots):
    c = pg.PlotCurveItem(pen=(i,nPlots*1.3))
    p_old.addItem(c)
    c.setPos(0,i*6)
    curves.append(c)

p_old.setYRange(0, nPlots * 6)
p_old.setXRange(0, nSamples)
p_old.resize(600, 900)

#rgn = pg.LinearRegionItem([nSamples/5., nSamples/3.])
#p.addItem(rgn)


data = np.random.normal(size=(nPlots*23, nSamples))
data1 = np.zeros((nPlots, nSamples))
ptr = 0
lastTime = time()
fps = None
count = 0



print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])


def update():
    global curve, data, ptr, p_old, lastTime, fps, nPlots, count
    count += 1
    #print "---------", count

    sample, timestamp = inlet.pull_sample(timeout=0.0000001)
    data1[:, :-1] = data1[:, 1:]
    data1[:, -1] = sample[:nPlots]

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
timer.start(0)



## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
