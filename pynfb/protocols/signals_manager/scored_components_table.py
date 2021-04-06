import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget

from ...protocols.ssd.topomap_canvas import TopographicMapCanvas


class BarLabelWidget(QtWidgets.QWidget):
    def __init__(self, value, max_value, min_value=0):
        super(BarLabelWidget, self).__init__()
        self.max_value = max_value
        self.min_value = min_value
        self.value = value

    def set_values(self, value, max_value, min_value=0):
        self.max_value = max_value
        self.min_value = min_value
        self.value = value

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        self.draw_value(e, qp)
        qp.end()

    def draw_value(self, event, qp):
        size = self.size()
        qp.setPen(QtCore.Qt.white)
        qp.setBrush(QtGui.QColor(51, 152, 188, 50))
        padding = 50 if 50 < size.height() else 0
        qp.drawRect(0, 0 + padding,
                    int(size.width() * (self.value - self.min_value) / (self.max_value - self.min_value)) - 1,
                    size.height() - 2 * padding - 1)
        qp.setPen(QtCore.Qt.black)
        qp.drawText(1, size.height()//2 + 1, str(round(self.value, 5)))


class TopoFilterCavas(QtWidgets.QWidget):
    def __init__(self, parent, names, topo, filter, size):
        super(TopoFilterCavas, self).__init__(parent)

        # topography layout
        topo_canvas = TopographicMapCanvas()
        topo_canvas.setMaximumWidth(size)
        topo_canvas.setMaximumHeight(size)
        topo_canvas.update_figure(topo, names=names, show_names=[], show_colorbar=False)

        # filter layout
        filter_canvas = TopographicMapCanvas()
        filter_canvas.setMaximumWidth(size)
        filter_canvas.setMaximumHeight(size)
        filter_canvas.update_figure(filter, names=names, show_names=[], show_colorbar=False)
        filter_canvas.setHidden(True)

        # layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(topo_canvas)
        layout.addWidget(filter_canvas)

        # attr
        self.show_filter = False
        self.topo = topo_canvas
        self.filter = filter_canvas
        self.names = names


    def switch(self):
        self.show_filter = not self.show_filter
        self.filter.setHidden(not self.show_filter)
        self.topo.setHidden(self.show_filter)

    def update_data(self, topo, filter):
        self.filter.update_figure(filter, names=self.names, show_names=[], show_colorbar=False)
        self.topo.update_figure(topo, names=self.names, show_names=[], show_colorbar=False)


class ScoredComponentsTable(QtWidgets.QTableWidget):
    one_selected = QtCore.pyqtSignal()
    more_one_selected = QtCore.pyqtSignal()
    no_one_selected = QtCore.pyqtSignal()

    def __init__(self, time_series, topographies, filters, channel_names, fs, scores, scores_name='Mutual info', marks=None, *args):
        super(ScoredComponentsTable, self).__init__(*args)

        # attributes
        self.row_items_max_height = 125
        self.time_series = time_series
        self.marks = marks
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
            checkbox = QtWidgets.QCheckBox()
            self.checkboxes.append(checkbox)
            self.setCellWidget(ind, self.columns.index('Selection'), checkbox)

            # topographies and filters
            topo_filter = TopoFilterCavas(self, self.channel_names, topographies[:, ind], filters[:, ind],
                                          self.row_items_max_height)
            self.topographies_items.append(topo_filter)
            self.setCellWidget(ind, self.columns.index('Topography'), topo_filter)

            # plots
            plot_widget = PlotWidget(enableMenu=False)
            if _previous_plot_link is not None:
                plot_widget.setXLink(_previous_plot_link)
                # plot_widget.setYLink(_previous_plot_link)
            _previous_plot_link = plot_widget
            x = np.arange(self.time_series.shape[0]) / fs
            plot_widget.plot(x=x, y=self.time_series[:, ind])
            if self.marks is not None:
                plot_widget.plot(x=x, y=self.marks*np.max(self.time_series[:, ind]), pen=(1,3))
                plot_widget.plot(x=x, y=-self.marks * np.max(self.time_series[:, ind]), pen=(1, 3))

            plot_widget.setMaximumHeight(self.row_items_max_height)
            plot_widget.plotItem.getViewBox().state['wheelScaleFactor'] = 0
            self.plot_items.append(plot_widget)
            self.setCellWidget(ind, 3, plot_widget)

            # scores
            score_widget = BarLabelWidget(scores[ind], max(scores), min(scores))
            self.scores.append(score_widget)
            self.setCellWidget(ind, self.columns.index(scores_name), score_widget)

        # formatting
        self.current_row = None
        self.horizontalHeader().setStretchLastSection(True)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

        # clickable 3 column header
        self.horizontalHeader().sectionClicked.connect(self.handle_header_click)
        self.is_spectrum_mode = False

        # set scores and order
        self.set_scores(scores)

        # checkbox signals
        for checkbox in self.checkboxes:
            checkbox.stateChanged.connect(self.checkboxes_state_changed)

        # selection context menu
        header = self.horizontalHeader()
        header.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self.handle_header_menu)

        # ctrl+a short cut
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_A), self).activated.connect(self.ctrl_plus_a_event)

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
        menu = QtWidgets.QMenu()
        for name, method in zip(['Revert selection', 'Select all', 'Clear selection'],
                                [self.revert_selection, self.select_all, self.clear_selection]):
            action = QtWidgets.QAction(name, self)
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
        if index == 1:
            self.order = self.order[::-1]
            self.reorder()
        if index == 2:
            self.switch_topo_filter()

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
                x = np.arange(self.time_series.shape[0]) / self.fs
                plot_item.plot(x=x, y=y, clear=True)
                if self.marks is not None:
                    plot_item.plot(x=x, y=self.marks * np.max(self.time_series[:, ind]), pen=(1, 3))
                    plot_item.plot(x=x, y=-self.marks * np.max(self.time_series[:, ind]), pen=(1, 3))
                self.columns[-1] = 'Time series'
        self.setHorizontalHeaderLabels(self.columns)
        self.plot_items[-1].autoRange()
        if flag:
            self.plot_items[-1].setXRange(0, 60)

    def get_checked_rows(self):
        return [j for j, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()]

    def get_unchecked_rows(self):
        return [j for j, checkbox in enumerate(self.checkboxes) if not checkbox.isChecked()]

    def switch_topo_filter(self):
        self.columns[2] = 'Filters' if self.columns[2] == 'Topography' else 'Topography'
        self.setHorizontalHeaderLabels(self.columns)
        for topo_filt in self.topographies_items:
            topo_filt.switch()

    def reorder(self):
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
        max_score = max(scores)
        min_score = min(scores)
        for j, score in enumerate(self.scores):
            score.set_values(scores[j], max_score, min_score)
        self.order = np.argsort(scores)[::-1]
        self.reorder()

    def redraw(self, time_series, topographies, filters, scores):
        # components
        self.time_series = time_series
        self.set_spectrum_mode()

        for ind in range(self.rowCount()):
            widget = self.cellWidget(ind, 2)
            widget.update_data(topographies[:, ind], filters[:, ind])
            self.topographies_items[ind] = widget

        # scores
        self.set_scores(scores)
