import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.linalg import solve


class Topomap:
    def __init__(self, pos, res=64):
        xmin = pos[:, 0].min()
        xmax = pos[:, 0].max()
        ymin = pos[:, 1].min()
        ymax = pos[:, 1].max()
        x = pos[:, 0]
        y = pos[:, 1]
        xi = np.linspace(xmin, xmax, res)
        yi = np.linspace(ymin, ymax, res)
        xi, yi = np.meshgrid(xi, yi)
        xy = x.ravel() + y.ravel() * -1j
        d = xy[None, :] * np.ones((len(xy), 1))
        d = np.abs(d - d.T)
        n = d.shape[0]
        d.flat[::n + 1] = 1.
        g = (d * d) * (np.log(d) - 1.)
        g.flat[::n + 1] = 0.
        self.g_solver = g
        m, n = xi.shape
        xy = xy.T
        self.g_tensor = np.empty((m, n, xy.shape[0]))
        g = np.empty(xy.shape)
        for i in range(m):
            for j in range(n):
                d = np.abs(xi[i, j] + -1j * yi[i, j] - xy)
                mask = np.where(d == 0)[0]
                if len(mask):
                    d[mask] = 1.
                np.log(d, out=g)
                g -= 1.
                g *= d * d
                if len(mask):
                    g[mask] = 0.
                self.g_tensor[i, j] = g

    def get_topomap(self, v):
        weights = solve(self.g_solver, v.ravel())
        return self.g_tensor.dot(weights)


class TopomapWidget(pg.PlotWidget):
    def __init__(self, pos, res=64, parent=None):
        super(TopomapWidget, self).__init__(parent)
        self.topomap = Topomap(pos, res=res)
        self.img = pg.ImageItem()
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.getPlotItem().getAxis('left').hide()
        self.getPlotItem().getAxis('bottom').hide()
        self.setAspectLocked(True)
        self.addItem(self.img)
        self.setMaximumWidth(res)
        self.setMaximumHeight(res)
        coef = 0.82
        radius = int(res * coef)
        shift = int(res * (1 - coef)/ 2)
        self.setMask(QtGui.QRegion(QtCore.QRect(shift, shift, radius, radius), QtGui.QRegion.Ellipse))

    def set_topomap(self, data):
        tmap = self.topomap.get_topomap(data)
        self.img.setImage(tmap.T)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    pos = np.random.normal(size=(21, 2))
    ww = QtWidgets.QWidget()
    ll = QtWidgets.QHBoxLayout(ww)
    tmap_widget = TopomapWidget(pos, 100)
    ll.addWidget(tmap_widget)
    btn = QtWidgets.QPushButton('next')
    btn.clicked.connect(lambda : tmap_widget.set_topomap(np.random.normal(size=(21, 1))))
    ll.addWidget(btn)
    ww.show()
    app.exec_()
