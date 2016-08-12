from PyQt4 import QtGui, QtCore
from pynfb.protocols.ssd.topomap_canvas import TopographicMapCanvas
from pyqtgraph import PlotWidget, PlotItem
import sys

from pynfb.protocols import SelectSSDFilterWidget
from pynfb.protocols.user_inputs import SelectCSPFilterWidget
from pynfb.widgets.spatial_filter_setup import SpatialFilterSetup
from pynfb.signals import DerivedSignal
from numpy import dot

class Table(QtGui.QTableWidget):
    def __init__(self, time_series, topographies, channel_names, fs, *args):
        self.row_items_max_height = 125
        self.time_series = time_series
        self.topographies = topographies
        self.channel_names = channel_names
        self.fs = fs

        super(Table, self).__init__(*args)
        # set size and names
        self.columns = ['Select', 'Topography', 'Time series (push to switch mode)']
        self.setColumnCount(len(self.columns))
        self.setRowCount(len(time_series))
        self.setHorizontalHeaderLabels(self.columns)

        # columns widgets
        self.checkboxes = []
        self.topographies_items = []
        self.plot_items = []
        _previous_plot_link = None
        for ind in range(self.rowCount()):

            # checkboxes
            checkbox = QtGui.QCheckBox()
            self.checkboxes.append(checkbox)
            self.setCellWidget(ind, self.columns.index('Select'), checkbox)

            # topographies
            topo_canvas = TopographicMapCanvas()
            topo_canvas.setMaximumWidth(self.row_items_max_height)
            topo_canvas.setMaximumHeight(self.row_items_max_height)
            topo_canvas.update_figure(self.topographies[:, ind], names=self.channel_names, show_names=[],
                                      show_colorbar=False)
            self.topographies_items.append(topo_canvas)
            self.setCellWidget(ind, self.columns.index('Topography'), topo_canvas)

            # plots
            plot_widget = PlotWidget(enableMenu=False)
            if _previous_plot_link is not None:
                plot_widget.setXLink(_previous_plot_link)
                plot_widget.setYLink(_previous_plot_link)
            _previous_plot_link = plot_widget
            plot_widget.plot(x=np.arange(self.time_series.shape[1])/fs)
            plot_widget.plot(y=self.time_series[ind])
            plot_widget.setMaximumHeight(self.row_items_max_height)
            self.plot_items.append(plot_widget)
            self.setCellWidget(ind, 2, plot_widget)

        # formatting
        self.current_row = None
        self.horizontalHeader().setStretchLastSection(True)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

        # clickable 3 column
        self.horizontalHeader().sectionClicked.connect(self.changeHorizontalHeader)
        self.is_spectrum_mode = False

    def changeHorizontalHeader(self, index):
        if index == 2:
            self.set_spectrum_mode(flag=not self.is_spectrum_mode)

    def set_spectrum_mode(self, flag=False):
        self.is_spectrum_mode = flag
        for ind, plot_item in enumerate(self.plot_items):
            y = self.time_series[ind]
            if flag:
                y = np.abs(np.fft.fft(y) / y.shape[0])
                x = np.fft.fftfreq(y.shape[-1], d=1/self.fs)
                plot_item.plot(x=x[:y.shape[0]//2], y=y[:y.shape[0]//2], clear=True)
                self.columns[-1] = 'Spectrum'
            else:
                plot_item.plot(x=np.arange(self.time_series.shape[1]) / fs, y=y, clear=True)
                self.columns[-1] = 'Time series'
        self.plot_items[-1].autoRange()
        self.setHorizontalHeaderLabels(self.columns)




class ICADialog(QtGui.QWidget):
    def __init__(self, time_series, topographies, channel_names, fs, parent=None):
        super(ICADialog, self).__init__(parent)
        # layout
        layout = QtGui.QVBoxLayout(self)
        table = Table(time_series, topographies, channel_names, fs)
        layout.addWidget(table)

if __name__ == '__main__':
    import numpy as np
    app = QtGui.QApplication([])
    n_channels = 3
    fs = 100
    from pynfb.widgets.helpers import ch_names_to_2d_pos
    channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9', 'Fc5', 'Fc1', 'Fc2', 'Fc6', 'Ft10', 'T7', 'C3', 'Cz',
                'C4', 'T8', 'Tp9', 'Cp5', 'Cp1', 'Cp2', 'Cp6', 'Tp10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    channels = channels[:n_channels]

    x = np.array([np.sin(10*(f+1)*2*np.pi*np.arange(0, 10, 1/fs)) for f in range(n_channels)])
    w = ICADialog(x + 0*np.random.randn(n_channels, x.shape[1]), np.random.randn(n_channels, n_channels), channels, fs)
    w.show()
    app.exec_()