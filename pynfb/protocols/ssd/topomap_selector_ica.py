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
    def __init__(self, time_series, topographies, channel_names, *args):
        self.row_items_max_height = 150

        self._for_link = None
        self.time_series = time_series
        self.topographies = topographies
        self.channel_names = channel_names

        super(Table, self).__init__(*args)
        # set size and names
        self.columns = ['Select', 'Topography', 'Time-space']
        self.setColumnCount(len(self.columns))
        self.setRowCount(len(time_series))
        self.setHorizontalHeaderLabels(self.columns)

        # buttons
        self.checkboxes = []
        for j in range(self.rowCount()):
            checkbox = QtGui.QCheckBox()
            self.checkboxes.append(checkbox)
            self.setCellWidget(j, self.columns.index('Select'), checkbox)
            self.update_row(j)

        # formatting
        self.current_row = None
        self.horizontalHeader().setStretchLastSection(True)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def update_row(self, ind, modified=False):

        # topography
        topo_canvas = TopographicMapCanvas()
        topo_canvas.setMaximumWidth(self.row_items_max_height)
        topo_canvas.setMaximumHeight(self.row_items_max_height)
        topo_canvas.update_figure(self.topographies[:, ind], names=self.channel_names, show_names=[], show_colorbar=False)
        self.setCellWidget(ind, self.columns.index('Topography'), topo_canvas)

        # plot widget
        plot_widget = PlotWidget(enableMenu=False)
        if self._for_link is not None:
            plot_widget.setXLink(self._for_link)
            plot_widget.setYLink(self._for_link)
        self._for_link = plot_widget
        plot_widget.plot(y=self.time_series[ind])
        plot_widget.setMaximumHeight(self.row_items_max_height)

        plot_widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setCellWidget(ind, self.columns.index('Time-space'), plot_widget)



if __name__ == '__main__':
    import numpy as np
    app = QtGui.QApplication([])
    n_channels = 32
    x = np.random.rand(10000, n_channels)
    from pynfb.widgets.helpers import ch_names_to_2d_pos
    channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9', 'Fc5', 'Fc1', 'Fc2', 'Fc6', 'Ft10', 'T7', 'C3', 'Cz',
                'C4', 'T8', 'Tp9', 'Cp5', 'Cp1', 'Cp2', 'Cp6', 'Tp10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    channels = channels[:n_channels]

    w = Table(np.random.randn(n_channels, 10000), np.random.randn(n_channels, n_channels), channels)
    w.show()
    app.exec_()