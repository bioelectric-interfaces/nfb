import numpy as np
from PyQt4 import QtGui, QtCore
from pyqtgraph import PlotWidget

from pynfb.protocols.ssd.topomap_canvas import TopographicMapCanvas


class ScoredComponentsTable(QtGui.QTableWidget):
    one_selected = QtCore.pyqtSignal()
    more_one_selected = QtCore.pyqtSignal()
    no_one_selected = QtCore.pyqtSignal()

    def __init__(self, time_series, topographies, channel_names, fs, scores, scores_name='Mutual info', *args):
        super(ScoredComponentsTable, self).__init__(*args)

        # attributes
        self.row_items_max_height = 125
        self.time_series = time_series
        self.topographies = topographies
        self.channel_names = channel_names
        self.fs = fs

        # set size and names
        self.columns = ['Selection', scores_name, 'Topography', 'Time series (push to switch mode)']
        self.setColumnCount(len(self.columns))
        self.setRowCount(time_series.shape[1])
        self.setHorizontalHeaderLabels(self.columns)

        # columns widgets
        self.checkboxes = []
        self.topographies_items = []
        self.plot_items = []
        self.scores = []
        _previous_plot_link = None
        for ind in range(self.rowCount()):

            # checkboxes
            checkbox = QtGui.QCheckBox()
            self.checkboxes.append(checkbox)
            self.setCellWidget(ind, self.columns.index('Selection'), checkbox)

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
            plot_widget.plot(x=np.arange(self.time_series.shape[0]) / fs)
            plot_widget.plot(y=self.time_series[:, ind])
            plot_widget.setMaximumHeight(self.row_items_max_height)
            plot_widget.plotItem.getViewBox().state['wheelScaleFactor'] = 0
            self.plot_items.append(plot_widget)
            self.setCellWidget(ind, 3, plot_widget)

            # scores
            checkbox = QtGui.QLabel(str(scores[ind]))
            self.scores.append(checkbox)
            self.setCellWidget(ind, self.columns.index('Mutual info'), checkbox)

        # formatting
        self.current_row = None
        self.horizontalHeader().setStretchLastSection(True)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

        # clickable 3 column header
        self.horizontalHeader().sectionClicked.connect(self.handle_header_click)
        self.is_spectrum_mode = False

        # reorder
        self.order = np.argsort(scores)
        self.reorder()

        # checkbox signals
        for checkbox in self.checkboxes:
            checkbox.stateChanged.connect(self.checkboxes_state_changed)

        # selection context menu
        header = self.horizontalHeader()
        header.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self.handle_header_menu)

        # ctrl+a short cut
        self.connect(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_A), self),
                     QtCore.SIGNAL('activated()'), self.ctrl_plus_a_event)

        # checkbox cell clicked
        self.cellClicked.connect(self.cell_was_clicked)

    def cell_was_clicked(self, row, column):
        if column == 0:
            self.checkboxes[self.order[row]].click()

    def ctrl_plus_a_event(self):
        if len(self.get_unchecked_rows()) == 0:
            self.clear_selection()
        else:
            self.select_all()

    def contextMenuEvent(self, pos):
        if self.columnAt(pos.x()) == 0:
            self.checkboxes[self.order[self.rowAt(pos.y())]].click() # state will be unchanged
            self.open_selection_menu()

    def handle_header_menu(self, pos):
        if self.horizontalHeader().logicalIndexAt(pos) == 0:
            self.open_selection_menu()

    def open_selection_menu(self):
        menu = QtGui.QMenu()
        for name, method in zip(['Revert selection', 'Select all', 'Clear selection'],
                                [self.revert_selection, self.select_all, self.clear_selection]):
            action = QtGui.QAction(name, self)
            action.triggered.connect(method)
            menu.addAction(action)
        menu.exec_(QtGui.QCursor.pos())

    def revert_selection(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(not checkbox.isChecked())

    def select_all(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)

    def clear_selection(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)

    def checkboxes_state_changed(self):
        checked = self.get_checked_rows()
        if len(checked) == 0:
            self.no_one_selected.emit()
        elif len(checked) == 1:
            self.one_selected.emit()
        else:
            self.more_one_selected.emit()

    def handle_header_click(self, index):
        if index == 3:
            self.set_spectrum_mode(flag=not self.is_spectrum_mode)
        if index == self.columns.index('Mutual info'):
            self.reorder()

    def set_spectrum_mode(self, flag=False):
        self.is_spectrum_mode = flag
        for ind, plot_item in enumerate(self.plot_items):
            y = self.time_series[:, ind]
            if flag:
                window = int(4 * self.fs)
                if len(y) >= window:
                    y = np.mean([np.abs(np.fft.fft(y[j:j + window]) ** 2 / y.shape[0])
                                 for j in range(0, y.shape[0] - window, window // 2)], axis=0)
                else:
                    y = np.abs(np.fft.fft(y) / y.shape[0]) ** 2
                x = np.fft.fftfreq(window, d=1 / self.fs)
                plot_item.plot(x=x[:y.shape[0] // 2], y=y[:y.shape[0] // 2], clear=True)
                self.columns[-1] = 'Spectrum'
            else:
                plot_item.plot(x=np.arange(self.time_series.shape[0]) / self.fs, y=y, clear=True)
                self.columns[-1] = 'Time series'

        self.plot_items[-1].autoRange()
        if flag:
            self.plot_items[-1].setXRange(0, 60)

    def get_checked_rows(self):
        return [j for j, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()]

    def get_unchecked_rows(self):
        return [j for j, checkbox in enumerate(self.checkboxes) if not checkbox.isChecked()]

    def reorder(self, order=None):
        self.order = self.order[-1::-1] if order is None else order
        for ind, new_ind in enumerate(self.order):
            self.insertRow(ind)
            self.setCellWidget(ind, 0, self.checkboxes[new_ind])
            self.setCellWidget(ind, 2, self.topographies_items[new_ind])
            self.setCellWidget(ind, 3, self.plot_items[new_ind])
            self.setCellWidget(ind, 1, self.scores[new_ind])
            self.setRowHeight(ind, self.row_items_max_height)
        for ind in range(self.rowCount() // 2):
            self.removeRow(self.rowCount() - 1)

    def set_scores(self, scores):
        for j, score in enumerate(self.scores):
            score.setText(str(scores[j]))
        self.order = np.argsort(scores)
        self.reorder()