from PyQt4 import QtGui, QtCore
from pynfb.protocols.ssd.topomap_canvas import TopographicMapCanvas
from pyqtgraph import PlotWidget
from mne import create_info
from mne.io import RawArray
from mne.channels import read_montage
from mne.preprocessing import ICA
import numpy as np
from sklearn.metrics import mutual_info_score

def mutual_info(x, y, bins=None):
    bins = len(x) // 5 if bins is None else bins
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

class Table(QtGui.QTableWidget):
    def __init__(self, time_series, topographies, channel_names, fs, scores=None,*args):
        self.row_items_max_height = 125
        self.time_series = time_series
        self.topographies = topographies
        self.channel_names = channel_names
        self.fs = fs
        self.order = list(range(len(self.channel_names))) if scores is None else np.argsort(scores)

        super(Table, self).__init__(*args)
        # set size and names
        self.columns = ['Select', 'Mutual info', 'Topography', 'Time series (push to switch mode)']
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
            plot_widget.plot(x=np.arange(self.time_series.shape[0])/fs)
            plot_widget.plot(y=self.time_series[:, ind])
            plot_widget.setMaximumHeight(self.row_items_max_height)
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

        # clickable 3 column
        self.horizontalHeader().sectionClicked.connect(self.changeHorizontalHeader)
        self.is_spectrum_mode = False

        # reorder
        self.reorder()

    def changeHorizontalHeader(self, index):
        if index == 3:
            self.set_spectrum_mode(flag=not self.is_spectrum_mode)
        if index == self.columns.index('Mutual info'):
            self.reorder()

    def set_spectrum_mode(self, flag=False):
        self.is_spectrum_mode = flag
        for ind, plot_item in enumerate(self.plot_items):
            y = self.time_series[:, ind]
            if flag:
                y = np.abs(np.fft.fft(y) / y.shape[0])
                x = np.fft.fftfreq(y.shape[-1], d=1/self.fs)
                plot_item.plot(x=x[:y.shape[0]//2], y=y[:y.shape[0]//2], clear=True)
                self.columns[-1] = 'Spectrum'
            else:
                plot_item.plot(x=np.arange(self.time_series.shape[0]) / fs, y=y, clear=True)
                self.columns[-1] = 'Time series'
        self.plot_items[-1].autoRange()
        self.setHorizontalHeaderLabels(self.columns)

    def get_checked_rows(self):
        return [j for j, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()]

    def reorder(self, order=None):
        self.order = self.order[-1::-1] if order is None else order
        for ind, new_ind in enumerate(self.order):
            self.insertRow(ind)
            self.setCellWidget(ind, 0, self.checkboxes[new_ind])
            self.setCellWidget(ind, 2, self.topographies_items[new_ind])
            self.setCellWidget(ind, 3, self.plot_items[new_ind])
            self.setCellWidget(ind, 1, self.scores[new_ind])
            self.setRowHeight(ind, self.row_items_max_height)
        for ind in range(self.rowCount()//2):
            self.removeRow(self.rowCount()-1)

    def set_scores(self, scores):
        for j, score in enumerate(self.scores):
            score.setText(str(scores[j]))
        self.order = np.argsort(scores)
        self.reorder()




class ICADialog(QtGui.QDialog):
    def __init__(self, raw_data, channel_names, fs, parent=None):
        super(ICADialog, self).__init__(parent)

        # attributes
        self.data = raw_data
        self.channel_names = channel_names
        self.n_channels = len(channel_names)
        self.rejection = None

        # unmixing matrix estimation
        raw_inst = RawArray(self.data.T, create_info(channel_names, fs, 'eeg', None))
        ica = ICA(method='extended-infomax')
        ica.fit(raw_inst)
        self.unmixing_matrix = np.dot(ica.unmixing_matrix_, ica.pca_components_[:ica.n_components_]).T
        self.components = np.dot(self.data, self.unmixing_matrix)

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
        self.sort_combo.setCurrentIndex(fp1_or_fp2_index)
        self.sort_combo.currentIndexChanged.connect(self.sort_by_mutual)
        sort_layout.addWidget(QtGui.QLabel('Sort by: '))
        sort_layout.addWidget(self.sort_combo)
        sort_layout.setAlignment(QtCore.Qt.AlignLeft)

        # mutual sorting
        scores = [mutual_info(self.components[:, j], self.data[:, fp1_or_fp2_index])
                  for j in range(self.components.shape[1])]

        # table
        # table = Table(ica.get_sources(raw_inst).to_data_frame().as_matrix(), self.unmixing_matrix, channel_names, fs)
        self.table = Table(self.components, self.unmixing_matrix, channel_names, fs, scores)

        # reject selected button
        self.reject_button = QtGui.QPushButton('Reject and Close')
        self.reject_button.setMaximumWidth(100)
        self.reject_button.clicked.connect(self.reject_and_close)


        #self.sort_button.clicked.connect(lambda : self.table.reorder())

        # layout
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(sort_layout)
        layout.addWidget(self.reject_button)

        # enable maximize btn
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMaximizeButtonHint )

    def sort_by_mutual(self):
        ind = self.sort_combo.currentIndex()
        scores = [mutual_info(self.components[:, j], self.data[:, ind]) for j in range(self.components.shape[1])]
        self.table.set_scores(scores)

    def reject_and_close(self):
        unmixing_matrix = self.unmixing_matrix.copy()
        inv = np.linalg.pinv(self.unmixing_matrix)
        unmixing_matrix[:, self.table.get_checked_rows()] = 0
        self.rejection = np.dot(unmixing_matrix, inv)
        self.close()

    @classmethod
    def get_rejection(cls, raw_data, channel_names, fs):
        selector = cls(raw_data, channel_names, fs)
        result = selector.exec_()
        return selector.rejection

if __name__ == '__main__':
    import numpy as np
    app = QtGui.QApplication([])
    n_channels = 3
    fs = 100
    from pynfb.widgets.helpers import ch_names_to_2d_pos
    channels = ['Cp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9', 'Fc5', 'Fc1', 'Fc2', 'Fc6', 'Ft10', 'T7', 'C3', 'Cz',
                'C4', 'T8', 'Tp9', 'Cp5', 'Cp1', 'Cp2', 'Cp6', 'Tp10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    channels = channels[:n_channels]

    x = np.array([np.sin(10*(f+1)*2*np.pi*np.arange(0, 10, 1/fs)) for f in range(n_channels)]).T

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
        rejection = ICADialog.get_rejection(x, channels, fs)
        if rejection is not None:
            x = np.dot(x, rejection)