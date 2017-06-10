import numpy as np
from mne import create_info
from mne.io import RawArray
from mne.preprocessing import ICA

from pynfb.signal_processing.filters import SpatialFilter
from pynfb.signals.rejections import SpatialRejection
from pynfb.widgets.helpers import ch_names_to_2d_pos
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh, inv
from sklearn.metrics import mutual_info_score

DEFAULTS = {'bandpass_low': 3,
            'regularizator': 0.05,
            'bandpass_high': 45}
BAND_DEFAULT = (DEFAULTS['bandpass_low'], DEFAULTS['bandpass_high'])


def mutual_info(x, y, bins=100):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


class SpatialDecomposition:
    def __init__(self, channel_names, fs, band=None):
        self.channel_names = channel_names
        self.pos = ch_names_to_2d_pos(channel_names)
        self.fs = fs
        self.band = band if band else BAND_DEFAULT
        self.filters = None  # un-mixing matrix
        self.topographies = None  # transposed mixing matrix
        self.scores = None  # eigenvalues (squared de-synchronization)
        self.name = None

    def fit(self, X, y=None):
        Wn = [self.band[0] / (0.5 * self.fs), self.band[1] / (0.5 * self.fs)]
        b, a = butter(4, Wn, btype='bandpass')
        X = filtfilt(b, a, X, axis=0)
        self.scores, self.filters, self.topographies = self.decompose(X, y)

    def decompose(self, X, y=None):
        raise NotImplementedError

    def set_parameters(self, **parameters):
        self.band = (parameters['bandpass_low'], parameters['bandpass_high'])

    def get_filter(self, index=None):
        """
        Return spatial filter
        :param index:
        :return:
        """
        index = index or slice(None)
        filter_ = SpatialFilter(self.filters[:, index], self.topographies[:, index])
        return filter_

    def get_rejections(self, indexes):
        unmixing_matrix = self.filters.copy()
        inv = np.linalg.pinv(unmixing_matrix)
        unmixing_matrix[:, indexes] = 0
        self.rejection = SpatialRejection(np.dot(unmixing_matrix, inv), rank=len(indexes), type_str=self.name,
                                          topographies=self.topographies[:, indexes])


class CSPDecomposition(SpatialDecomposition):
    def __init__(self, channel_names, fs, band=None, reg_coef=0.001):
        super(CSPDecomposition, self).__init__(channel_names, fs, band)
        self.reg_coef = reg_coef
        self.name = 'csp'

    def decompose(self, X, y=None):
        states = [X[~y.astype(bool)], X[y.astype(bool)]]
        covs = [np.dot(state.T, state) / state.shape[0] for state in states]

        # find filters
        regularization = lambda z: z + self.reg_coef * np.trace(z) * np.eye(z.shape[0]) / z.shape[0]
        vals, vecs = eigh(regularization(covs[0]), regularization(covs[1]))
        vecs /= np.abs(vecs).max(0)

        # return vals, vecs and topographics (in descending order)
        reversed_slice = slice(-1, None, -1)
        topo = inv(vecs[:, reversed_slice]).T
        return vals[reversed_slice], vecs[:, reversed_slice], topo

    def set_parameters(self, **parameters):
        super(CSPDecomposition, self).set_parameters(**parameters)
        self.reg_coef = parameters['regularizator']


class ICADecomposition(SpatialDecomposition):
    def __init__(self, channel_names, fs, band=None):
        super(ICADecomposition, self).__init__(channel_names, fs, band)
        self.sorted_channel_index = 0
        self.name = 'ica'

    def decompose(self, X, y=None):
        raw_inst = RawArray(X.T, create_info(self.channel_names, self.fs, 'eeg', None))
        ica = ICA(method='extended-infomax')
        ica.fit(raw_inst)
        filters = np.dot(ica.unmixing_matrix_, ica.pca_components_[:ica.n_components_]).T
        topographies = np.linalg.inv(filters).T
        scores = self.get_scores(X, filters)
        return scores, filters, topographies

    def get_scores(self, X, filters, ch_name=None, index=None):
        if index is None:
            index = np.argmax(self.pos[:, 1]) if ch_name is None else self.channel_names.index(ch_name)
        self.sorted_channel_index = index
        components = np.dot(X, filters)
        scores = [mutual_info(components[:, j], X[:, index]) for j in range(components.shape[1])]
        return scores


class SpatialDecompositionPool:
    def __init__(self, channel_names, fs, bands=None, dec_class='csp', indexes=None):
        Decomposition = {'ica': ICADecomposition, 'csp': CSPDecomposition}[dec_class]
        # [(k*2, k*2+4) for k in range(3, 11)]
        self.bands = bands or [(6, 10), (8, 12), (10, 14), (12, 16), (14, 18), (16, 20), (18, 22), (20, 24)]
        self.indexes = np.array(indexes or [1, -1])
        if self.indexes.ndim == 1:
            self.indexes = self.indexes[None, :].repeat(len(bands), 0)
        self.pool = [Decomposition(channel_names, fs, band) for band in bands]

    def fit(self, X, y=None):
        for decomposer in self.pool:
            decomposer.fit(X, y)

    def get_filter(self):
        filters = [dec.filters[:, index] for dec, index in zip(self.pool, self.indexes)]
        topographies = [dec.topographies[:, index] for dec, index in zip(self.pool, self.indexes)]
        return SpatialFilter(np.hstack(filters), np.hstack(topographies))

if __name__ == '__main__':
    dec = CSPDecomposition(['Fp1', 'Fp2'], 500)
    dec.decompose(None, None)
