import sys
from PyQt4 import QtGui, QtCore
import numpy as np
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
try:
    from mne.viz import plot_topomap
except ImportError:
    pass


class TopographicMapCanvas(FigureCanvas):
    def __init__(self, data, pos, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.colorbar = None
        FigureCanvas.__init__(self, self.fig)
        self.update_figure(data, pos)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def update_figure(self, data, pos):
        self.axes.clear()
        if self.colorbar:
            self.colorbar.remove()
        a, b = plot_topomap(data, pos, axes=self.axes, show=False, contours=0)
        self.colorbar = self.fig.colorbar(a, orientation='horizontal', ax = self.axes)
        self.colorbar.ax.tick_params(labelsize=6)
        self.colorbar.ax.set_xticklabels(self.colorbar.ax.get_xticklabels(), rotation=90)
        self.draw()

    def test_update_figure(self):
        data = np.random.randn(3)
        pos = np.array([(0, 0), (1, -1), (-1, -1)])
        self.update_figure(data=data, pos=pos)


if __name__ == '__main__':
    qApp = QtGui.QApplication(sys.argv)
    aw = TopographicMapCanvas(np.random.randn(3), np.array([(0, 0), (1, -1), (-1, -1)]), width=5, height=4, dpi=100)
    timer = QtCore.QTimer(qApp)
    timer.timeout.connect(aw.test_update_figure)
    timer.start(1000)
    aw.show()
    sys.exit(qApp.exec_())