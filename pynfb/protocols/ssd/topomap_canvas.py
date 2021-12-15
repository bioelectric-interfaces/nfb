import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from pynfb.widgets.helpers import ch_names_to_2d_pos
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def update_figure(self, data, pos=None, names=None, show_names=None, show_colorbar=True, central_text=None,
                      right_bottom_text=None, show_not_found_symbol=False, montage=None):
        if montage is None:
            if pos is None:
                pos = ch_names_to_2d_pos(names)
        else:
            pos = montage.get_pos('EEG')
            names = montage.get_names('EEG')
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
            data[0] += 0.1
            data[1] -= 0.1
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
        if central_text is not None:
            self.axes.text(0, 0, central_text, horizontalalignment='center', verticalalignment='center')

        if right_bottom_text is not None:
            self.axes.text(-0.65, 0.65, right_bottom_text, horizontalalignment='left', verticalalignment='top')

        if show_not_found_symbol:
            self.axes.text(0, 0, '/', horizontalalignment='center', verticalalignment='center')
            self.axes.text(0, 0, 'O', size=10, horizontalalignment='center', verticalalignment='center')

        if show_colorbar:
            self.colorbar = self.fig.colorbar(a, orientation='horizontal', ax=self.axes)
            self.colorbar.ax.tick_params(labelsize=6)
            self.colorbar.ax.set_xticklabels(self.colorbar.ax.get_xticklabels(), rotation=90)
        self.draw()

    def draw_central_text(self, text='', right_bottom_text='', show_not_found_symbol=False):
        data = np.random.randn(3)*0
        pos = np.array([(0, 0), (1, -1), (-1, -1)])
        self.update_figure(data, pos, central_text=text, names=[], show_colorbar=False,
                           right_bottom_text=right_bottom_text, show_not_found_symbol=show_not_found_symbol)

    def test_update_figure(self):

        from pynfb.inlets.montage import Montage
        montage = Montage(names=['Fp1', 'Fp2', 'Cz', 'AUX', 'MEG 2632'])
        print(montage)
        data = np.random.randn(3)
        pos = np.array([(0, 0), (1, -1), (-1, -1)])
        self.update_figure(data=data, pos=pos, names=['c1', 'c2', 'oz'], montage=montage)



