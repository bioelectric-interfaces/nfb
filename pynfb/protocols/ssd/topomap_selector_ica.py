import h5py
from PyQt5 import QtCore, QtGui, QtWidgets
from mne import create_info
from mne.io import RawArray
from mne.preprocessing import ICA
from sklearn.metrics import mutual_info_score

from ...serializers.xml_ import get_lsl_info_from_xml
from ...postprocessing.helpers import dc_blocker
from ...protocols.signals_manager.scored_components_table import ScoredComponentsTable
import numpy as np
from ...protocols.ssd.sliders_csp import Sliders
from ...signal_processing.filters import SpatialRejection
from ...widgets.helpers import ch_names_to_2d_pos, WaitMessage
from ..._titles import WAIT_BAR_MESSAGES
from ...signal_processing.decompositions import CSPDecomposition, ICADecomposition, CSPDecompositionStimulus
from ...signal_processing.helpers import stimulus_split
from time import time

def mutual_info(x, y, bins=100):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


class ICADialog(QtWidgets.QDialog):
    def __init__(self, raw_data, channel_names, fs, parent=None, decomposition=None, mode='ica', filters=None,
                 scores=None, states=None, labels=None, _stimulus_split=False, marks=None, band=None):
        super(ICADialog, self).__init__(parent)
        self.setWindowTitle(mode.upper())
        self.setMinimumWidth(800)
        self.setMinimumHeight(400)


        if decomposition is None:
            if mode == 'csp':
                if not _stimulus_split:
                    self.decomposition = CSPDecomposition(channel_names, fs, band)
                    if labels is None:
                        labels = np.zeros(raw_data.shape[0])
                        labels[len(labels)//2:] = 1
                else:
                    self.decomposition = CSPDecompositionStimulus(channel_names, fs, band)
                    labels = marks
            elif mode == 'ica':
                self.decomposition = ICADecomposition(channel_names, fs, band)
        else:
            self.decomposition = decomposition

        # attributes
        self.sampling_freq = fs
        self.rejection = None
        self.spatial = None
        self.topography = None
        self.bandpass = None
        self.table = None
        self.mode = mode
        self.raw_data = raw_data
        self.labels = labels
        self.data = self.raw_data

        # unmixing matrix estimation
        timer = time()
        if decomposition is None:
            self.decomposition.fit(self.raw_data, self.labels)
        self.scores = self.decomposition.scores
        self.unmixing_matrix = self.decomposition.filters
        self.topographies = self.decomposition.topographies
        self.components = np.dot(self.decomposition.temporal_filter.apply(self.raw_data), self.unmixing_matrix)


        print('ICA/CSP time elapsed = {}s'.format(time() - timer))
        timer = time()


        scores_name = 'Mutual info' if mode == 'ica' else 'Eigenvalues'
        # table
        self.table = ScoredComponentsTable(self.components, self.topographies, self.unmixing_matrix, channel_names, fs,
                                           self.scores, scores_name=scores_name, marks=marks if _stimulus_split else None)
        print('Table drawing time elapsed = {}s'.format(time() - timer))

        # reject selected button
        self.reject_button = QtWidgets.QPushButton('Reject selection')
        self.spatial_button = QtWidgets.QPushButton('Make spatial filter')
        self.add_to_all_checkbox = QtWidgets.QCheckBox('Add to all signals')
        self.reject_button.setMaximumWidth(150)
        self.spatial_button.setMaximumWidth(150)
        self.reject_button.clicked.connect(self.reject_and_close)
        self.spatial_button.clicked.connect(self.spatial_and_close)

        # layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table)
        self.update_band_checkbox = QtWidgets.QCheckBox('Update band')

        # setup sliders
        self.sliders = Sliders(fs, reg_coef=(mode == 'csp'), stimulus_split=_stimulus_split, band=band)
        self.sliders.apply_button.clicked.connect(self.recompute)
        self.lambda_csp3 = states
        layout.addWidget(self.sliders)
        layout.addWidget(self.update_band_checkbox)

        # ica mutual sorting
        if mode == 'ica':
            sort_layout = QtWidgets.QHBoxLayout()
            self.sort_combo = QtWidgets.QComboBox()
            self.sort_combo.setMaximumWidth(100)
            self.sort_combo.addItems(channel_names)
            self.sort_combo.setCurrentIndex(self.decomposition.sorted_channel_index)
            self.sort_combo.currentIndexChanged.connect(self.sort_by_mutual)
            sort_layout.addWidget(QtWidgets.QLabel('Sort by: '))
            sort_layout.addWidget(self.sort_combo)
            sort_layout.setAlignment(QtCore.Qt.AlignLeft)
            layout.addLayout(sort_layout)

        # buttons
        buttons_layout = QtWidgets.QHBoxLayout()
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
        self.rejection = SpatialRejection(np.dot(unmixing_matrix, inv), rank=len(indexes), type_str=self.mode,
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
        self.decomposition.set_parameters(**parameters)
        self.decomposition.fit(self.raw_data, self.labels)
        self.scores = self.decomposition.scores
        self.unmixing_matrix = self.decomposition.filters
        self.topographies = self.decomposition.topographies
        self.components = np.dot(self.decomposition.temporal_filter.apply(self.raw_data), self.unmixing_matrix)
        self.table.redraw(self.components, self.topographies, self.unmixing_matrix, self.scores)

    @classmethod
    def get_rejection(cls, raw_data, channel_names, fs, decomposition=None, mode='ica', states=None, labels=None,
                      _stimulus_split=False, marks=None, band=None):
        wait_bar = WaitMessage(mode.upper() + WAIT_BAR_MESSAGES['CSP_ICA']).show_and_return()
        selector = cls(raw_data, channel_names, fs, decomposition=decomposition, mode=mode, states=states,
                       labels=labels, _stimulus_split=_stimulus_split, marks=marks, band=band)
        wait_bar.close()
        result = selector.exec_()
        bandpass = selector.bandpass if selector.update_band_checkbox.isChecked() else None
        return (selector.rejection,
                selector.spatial, selector.topography,
                selector.decomposition,
                bandpass,
                selector.add_to_all_checkbox.isChecked())


if __name__ == '__main__':
    import numpy as np

    app = QtWidgets.QApplication([])
    n_channels = 3
    fs = 100

    channels = ['Cp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9', 'Fc5', 'Fc1', 'Fc2', 'Fc6', 'Ft10', 'T7', 'C3', 'Cz',
                'C4', 'T8', 'Tp9', 'Cp5', 'Cp1', 'Cp2', 'Cp6', 'Tp10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    channels = channels[:n_channels]

    x = np.array([np.sin(10 * (f + 1) * 2 * np.pi * np.arange(0, 10, 1 / fs)) for f in range(n_channels)]).T

    # Generate sample data
    np.random.seed(0)
    n_samples = 2000
    t = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * t)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * t))  # Signal 2 : square signal
    from scipy import signal

    s3 = signal.sawtooth(2 * np.pi * t)  # Signal 3: saw tooth signal

    S = np.c_[s1, s2, s3]
    S += 0.1 * np.random.normal(size=S.shape)  # Add noise

    S /= S.std(axis=0)  # Standardize data
    # Mix data
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    x = np.dot(S, A.T)  # Generate observations
    y = np.ones(len(x))
    y[:len(y)//2] = 0


    if False:
        dir_ = 'D:\\vnd_spbu\\pilot\\mu5days'
        experiment = 'pilot5days_Skotnikova_Day4_03-02_13-33-55'
        with h5py.File('{}\\{}\\{}'.format(dir_, experiment, 'experiment_data.h5')) as f:
            ica = f['protocol1/signals_stats/left/rejections/rejection1'][:]

            x_filters = dc_blocker(np.dot(f['protocol1/raw_data'][:], ica))
            x_rotation = dc_blocker(np.dot(f['protocol2/raw_data'][:], ica))
            x_dict = {
                'closed': x_filters[:x_filters.shape[0] // 2],
                'opened': x_filters[x_filters.shape[0] // 2:],
                'rotate': x_rotation
            }
            x = np.concatenate([x_dict['closed'], x_dict['opened'], x_dict['rotate']])
            drop_channels = ['AUX', 'A1', 'A2']
            labels, fs = get_lsl_info_from_xml(f['stream_info.xml'][0])
            print('fs: {}\nall labels {}: {}'.format(fs, len(labels), labels))
            channels = [label for label in labels if label not in drop_channels]

    for j in range(4):
        rejection, spatial, unmixing = ICADialog.get_rejection(x, channels, fs, mode='ica', labels=y)
        if rejection is not None:
            x = np.dot(x, rejection)
