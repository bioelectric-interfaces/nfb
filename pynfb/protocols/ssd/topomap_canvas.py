import sys
from PyQt4 import QtGui, QtCore
import numpy as np
from pynfb.widgets.helpers import ch_names_to_2d_pos
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import rcParams
rcParams['font.size'] = 8
try:
    from mne.viz import plot_topomap
except ImportError:
    pass


class TopographicMapCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.colorbar = None
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def update_figure(self, data, pos=None, names=None, show_names=None, show_colorbar=True):
        if pos is None:
            pos = ch_names_to_2d_pos(names)
        data = np.array(data)
        self.axes.clear()
        if self.colorbar:
            self.colorbar.remove()
        if show_names is None:
            show_names = ['O1', 'O2', 'CZ', 'T3', 'T4', 'T7', 'T8', 'FP1', 'FP2']
        show_names = [name.upper() for name in show_names]
        mask = np.array([name.upper() in show_names for name in names]) if names else None
        v_min, v_max = None, None
        if (data == data[0]).all():
            data += np.random.uniform(-1e-3, 1e-3, size=len(data))
            v_min, v_max = -1, 1
        a, b = plot_topomap(data, pos, axes=self.axes, show=False, contours=0, names=names, show_names=True,
                            mask=mask,
                            mask_params=dict(marker='o',
                                             markerfacecolor='w',
                                             markeredgecolor='w',
                                             linewidth=0,
                                             markersize=3),
                            vmin=v_min,
                            vmax=v_max)
        if show_colorbar:
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