from PyQt4 import QtGui, QtCore
from mne import create_info
from mne.io import RawArray
from mne.preprocessing import ICA
from sklearn.metrics import mutual_info_score

from pynfb.protocols.signals_manager.scored_components_table import ScoredComponentsTable
import numpy as np
from pynfb.protocols.ssd.sliders_csp import Sliders
from pynfb.signals.rejections import Rejection
from pynfb.widgets.helpers import ch_names_to_2d_pos, WaitMessage
from pynfb._titles import WAIT_BAR_MESSAGES


def mutual_info(x, y, bins=100):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


class ICADialog(QtGui.QDialog):
    def __init__(self, raw_data, channel_names, fs, parent=None, unmixing_matrix=None, mode='ica', filters=None):
        super(ICADialog, self).__init__(parent)
        self.setWindowTitle(mode.upper())
        self.setMinimumWidth(800)
        self.setMinimumHeight(400)

        # attributes
        self.channel_names = channel_names
        self.pos = ch_names_to_2d_pos(channel_names)
        self.n_channels = len(channel_names)
        self.sampling_freq = fs
        self.rejection = None
        self.spatial = None
        self.topography = None
        self.bandpass = None
        self.table = None
        self.mode = mode

        # data preprocessing:
        from scipy.signal import butter, filtfilt
        b, a = butter(4, [0.5 / (0.5 * fs), 35 / (0.5 * fs)], btype='bandpass')
        self.data = filtfilt(b, a, raw_data, axis=0)

        if mode == 'csp':
            # Sliders
            self.sliders = Sliders()
            self.sliders.apply_button.clicked.connect(self.recompute)

        # unmixing matrix estimation
        self.unmixing_matrix = None
        self.topographies = None
        self.scores = None
        self.components = None
        from time import time
        timer = time()
        if unmixing_matrix is None:
            if mode == 'ica':
                raw_inst = RawArray(self.data.T, create_info(channel_names, fs, 'eeg', None))
                ica = ICA(method='extended-infomax')
                ica.fit(raw_inst)
                self.unmixing_matrix = np.dot(ica.unmixing_matrix_, ica.pca_components_[:ica.n_components_]).T
            # self.topographies = np.dot(ica.mixing_matrix_.T, ica.pca_components_[:ica.n_components_]).T
            elif mode == 'csp':
                self.recompute()
            else:
                raise TypeError('Wrong mode name')

        else:
            self.unmixing_matrix = unmixing_matrix
        self.topographies = np.linalg.inv(self.unmixing_matrix).T
        self.components = np.dot(self.data, self.unmixing_matrix)
        print('ICA/CSP time elapsed = {}s'.format(time() - timer))
        timer = time()

        if mode == 'ica':
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
            self.scores = [mutual_info(self.components[:, j], self.data[:, fp1_or_fp2_index])
                      for j in range(self.components.shape[1])]
            print('Mutual info scores time elapsed = {}s'.format(time() - timer))
            timer = time()

        scores_name = 'Mutual info' if mode == 'ica' else 'Eigenvalues'
        # table
        self.table = ScoredComponentsTable(self.components, self.topographies, channel_names, fs, self.scores,
                                           scores_name=scores_name)
        print('Table drawing time elapsed = {}s'.format(time() - timer))

        # reject selected button
        self.reject_button = QtGui.QPushButton('Reject selection')
        self.spatial_button = QtGui.QPushButton('Make spatial filter')
        self.add_to_all_checkbox = QtGui.QCheckBox('Add to all signals')
        self.reject_button.setMaximumWidth(150)
        self.spatial_button.setMaximumWidth(150)
        self.reject_button.clicked.connect(self.reject_and_close)
        self.spatial_button.clicked.connect(self.spatial_and_close)

        # layout
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.table)
        self.update_band_checkbox = QtGui.QCheckBox('Update band')
        if mode == 'csp':
            layout.addWidget(self.sliders)
            layout.addWidget(self.update_band_checkbox)
        if mode == 'ica':
            layout.addLayout(sort_layout)
        buttons_layout = QtGui.QHBoxLayout()
        buttons_layout.setAlignment(QtCore.Qt.AlignLeft)
        buttons_layout.addWidget(self.reject_button)
        buttons_layout.addWidget(self.spatial_button)
        buttons_layout.addWidget(self.add_to_all_checkbox)
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
        self.scores = [mutual_info(self.components[:, j], self.data[:, ind]) for j in range(self.components.shape[1])]
        self.table.set_scores(self.scores)

    def reject_and_close(self):
        indexes = self.table.get_checked_rows()
        unmixing_matrix = self.unmixing_matrix.copy()
        inv = np.linalg.pinv(self.unmixing_matrix)
        unmixing_matrix[:, indexes] = 0
        self.rejection = Rejection(np.dot(unmixing_matrix, inv), rank=len(indexes), type_str=self.mode,
                                   topographies=self.topographies[:, indexes])
        self.close()

    def spatial_and_close(self):
        index = self.table.get_checked_rows()[0]
        self.spatial =  self.unmixing_matrix[:, index]
        self.topography = self.topographies[:, index]
        print(index)
        self.close()

    def recompute(self):
        parameters = self.sliders.getValues()
        self.bandpass = (parameters['bandpass_low'], parameters['bandpass_high'])
        from pynfb.protocols.ssd.csp import csp
        self.scores, self.unmixing_matrix, self.topographies = csp(self.data,
                                                                   fs=self.sampling_freq,
                                                                   band=self.bandpass,
                                                                   regularization_coef=parameters['regularizator'])
        self.components = np.dot(self.data, self.unmixing_matrix)
        if self.table is not None:
            self.table.redraw(self.components, self.topographies, self.scores)

    @classmethod
    def get_rejection(cls, raw_data, channel_names, fs, unmixing_matrix=None, mode='ica'):
        wait_bar = WaitMessage(mode.upper() + WAIT_BAR_MESSAGES['CSP_ICA']).show_and_return()
        selector = cls(raw_data, channel_names, fs, unmixing_matrix=unmixing_matrix, mode=mode)
        wait_bar.close()
        result = selector.exec_()
        bandpass = selector.bandpass if selector.update_band_checkbox.isChecked() else None
        return (selector.rejection,
                selector.spatial, selector.topography,
                selector.unmixing_matrix,
                bandpass,
                selector.add_to_all_checkbox.isChecked())


if __name__ == '__main__':
    import numpy as np

    app = QtGui.QApplication([])
    n_channels = 3
    fs = 100

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
        rejection, spatial, unmixing = ICADialog.get_rejection(x, channels, fs)
        if rejection is not None:
            x = np.dot(x, rejection)
