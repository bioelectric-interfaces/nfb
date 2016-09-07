from PyQt4 import QtGui, QtCore
from pynfb.protocols.ssd.topomap_canvas import TopographicMapCanvas
from pyqtgraph import PlotWidget
from pyqtgraph import setConfigOption
from mne import create_info
from mne.io import RawArray
from mne.channels import read_montage
from mne.preprocessing import ICA
import numpy as np
from sklearn.metrics import mutual_info_score


def mutual_info(x, y, bins=100):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


class Table(QtGui.QTableWidget):
    one_selected = QtCore.pyqtSignal()
    more_one_selected = QtCore.pyqtSignal()
    no_one_selected = QtCore.pyqtSignal()

    def __init__(self, time_series, topographies, channel_names, fs, scores, *args):
        super(Table, self).__init__(*args)

        # attributes
        self.row_items_max_height = 125
        self.time_series = time_series
        self.topographies = topographies
        self.channel_names = channel_names
        self.fs = fs

        # set size and names
        self.columns = ['Selection', 'Mutual info', 'Topography', 'Time series (push to switch mode)']
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


class ICADialog(QtGui.QDialog):
    def __init__(self, raw_data, channel_names, fs, parent=None, ica=None):
        super(ICADialog, self).__init__(parent)
        self.setWindowTitle('ICA')
        self.setMinimumWidth(800)
        self.setMinimumHeight(400)

        # attributes
        self.data = raw_data
        self.channel_names = channel_names
        self.n_channels = len(channel_names)
        self.rejection = None

        # unmixing matrix estimation

        from time import time
        timer = time()
        raw_inst = RawArray(self.data.T, create_info(channel_names, fs, 'eeg', None))
        if ica is None:
            self.ica = ICA(method='extended-infomax')
            self.ica.fit(raw_inst)
            ica = self.ica
        else:
            self.ica = ica
        self.unmixing_matrix = np.dot(ica.unmixing_matrix_, ica.pca_components_[:ica.n_components_]).T
        self.topographies = np.dot(ica.mixing_matrix_.T, ica.pca_components_[:ica.n_components_]).T
        self.components = np.dot(self.data, self.unmixing_matrix)
        print('ICA time elapsed = {}s'.format(time() - timer))
        timer = time()
        # sort by fp1 or fp2
        sort_layout = QtGui.QHBoxLayout()
        self.sort_combo = QtGui.QComboBox()
        self.sort_combo.setMaximumWidth(100)
        self.sort_combo.addItems(channel_names)
        fp1_or_fp2_index = -1
        upper_channels_names = [ch.upper() for ch in channel_names]
        if 'FP1' in upper_channels_names:
            fp1_or_fp2_index = upper_channels_names.index('FP1')
        if fp1_or_fp2_index < 0 and 'FP2' in upper_channels_names:
            fp1_or_fp2_index = upper_channels_names.index('FP2')
        if fp1_or_fp2_index < 0:
            fp1_or_fp2_index = 0
        print('Sorting channel is', fp1_or_fp2_index)
        self.sort_combo.setCurrentIndex(fp1_or_fp2_index)
        self.sort_combo.currentIndexChanged.connect(self.sort_by_mutual)
        sort_layout.addWidget(QtGui.QLabel('Sort by: '))
        sort_layout.addWidget(self.sort_combo)
        sort_layout.setAlignment(QtCore.Qt.AlignLeft)

        # mutual sorting
        scores = [mutual_info(self.components[:, j], self.data[:, fp1_or_fp2_index])
                  for j in range(self.components.shape[1])]
        print('Mutual info scores time elapsed = {}s'.format(time() - timer))
        timer = time()
        # table
        # table = Table(ica.get_sources(raw_inst).to_data_frame().as_matrix(), self.unmixing_matrix, channel_names, fs)
        self.table = Table(self.components, self.topographies, channel_names, fs, scores)
        print('Table drawing time elapsed = {}s'.format(time() - timer))
        # reject selected button
        self.reject_button = QtGui.QPushButton('Reject selection')
        self.spatial_button = QtGui.QPushButton('Make spatial filter')
        self.reject_button.setMaximumWidth(150)
        self.spatial_button.setMaximumWidth(150)
        self.reject_button.clicked.connect(self.reject_and_close)
        self.spatial_button.clicked.connect(self.spatial_and_close)

        # self.sort_button.clicked.connect(lambda : self.table.reorder())

        # layout
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(sort_layout)
        buttons_layout = QtGui.QHBoxLayout()
        buttons_layout.setAlignment(QtCore.Qt.AlignLeft)
        buttons_layout.addWidget(self.reject_button)
        buttons_layout.addWidget(self.spatial_button)
        layout.addLayout(buttons_layout)

        # enable maximize btn
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMaximizeButtonHint)

        # checkboxes behavior
        self.table.no_one_selected.connect(lambda: self.reject_button.setDisabled(True))
        self.table.no_one_selected.connect(lambda: self.spatial_button.setDisabled(True))
        self.table.one_selected.connect(lambda: self.reject_button.setDisabled(False))
        self.table.one_selected.connect(lambda: self.spatial_button.setDisabled(False))
        self.table.more_one_selected.connect(lambda: self.reject_button.setDisabled(False))
        self.table.more_one_selected.connect(lambda: self.spatial_button.setDisabled(True))

        self.table.checkboxes_state_changed()

    def sort_by_mutual(self):
        ind = self.sort_combo.currentIndex()
        scores = [mutual_info(self.components[:, j], self.data[:, ind]) for j in range(self.components.shape[1])]
        self.table.set_scores(scores)

    def reject_and_close(self):
        indexes = self.table.get_checked_rows()
        unmixing_matrix = self.unmixing_matrix.copy()
        inv = np.linalg.pinv(self.unmixing_matrix)
        unmixing_matrix[:, indexes] = 0
        self.rejection = np.dot(unmixing_matrix, inv)
        self.close()

    def spatial_and_close(self):
        print('* Spatial')

    @classmethod
    def get_rejection(cls, raw_data, channel_names, fs, ica=None):
        selector = cls(raw_data, channel_names, fs, ica=ica)
        result = selector.exec_()
        return selector.rejection, selector.ica


if __name__ == '__main__':
    import numpy as np

    app = QtGui.QApplication([])
    n_channels = 3
    fs = 100
    from pynfb.widgets.helpers import ch_names_to_2d_pos

    channels = ['Cp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9', 'Fc5', 'Fc1', 'Fc2', 'Fc6', 'Ft10', 'T7', 'C3', 'Cz',
                'C4', 'T8', 'Tp9', 'Cp5', 'Cp1', 'Cp2', 'Cp6', 'Tp10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    channels = channels[:n_channels]

    x = np.array([np.sin(10 * (f + 1) * 2 * np.pi * np.arange(0, 10, 1 / fs)) for f in range(n_channels)]).T

    # Generate sample data
    np.random.seed(0)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    from scipy import signal

    s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

    S = np.c_[s1, s2, s3]
    S += 0.1 * np.random.normal(size=S.shape)  # Add noise

    S /= S.std(axis=0)  # Standardize data
    # Mix data
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    x = np.dot(S, A.T)  # Generate observations

    for j in range(4):
        rejection, _ = ICADialog.get_rejection(x, channels, fs)
        if rejection is not None:
            x = np.dot(x, rejection)
