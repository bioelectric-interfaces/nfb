import numpy as np
from mne import create_info
from mne.io import RawArray
from mne.preprocessing import ICA
import mne

from ..signal_processing.filters import SpatialFilter, ButterFilter, FilterSequence, FilterStack, SpatialRejection
from ..signal_processing.helpers import get_outliers_mask, stimulus_split
from ..widgets.helpers import ch_names_to_2d_pos
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh, inv
from sklearn.metrics import mutual_info_score
import packaging.version


DEFAULTS = {'bandpass_low': 3,
            'regularizator': 0.001,
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
        self.temporal_filter = ButterFilter(self.band, self.fs, len(self.channel_names))
        self.filters = None  # un-mixing matrix
        self.topographies = None  # transposed mixing matrix
        self.scores = None  # eigenvalues (squared de-synchronization)
        self.name = None
        self.outliers_mask = None

    def fit(self, X, y=None):
        X = self.temporal_filter.apply(X)
        self.outliers_mask = get_outliers_mask(X)
        good_mask = ~self.outliers_mask
        self.scores, self.filters, self.topographies = self.decompose(X[good_mask], y[good_mask] if y is not None else None)
        return self

    def decompose(self, X, y=None):
        raise NotImplementedError

    def set_parameters(self, **parameters):
        self.band = (parameters['bandpass_low'], parameters['bandpass_high'])
        self.temporal_filter = ButterFilter(self.band, self.fs, len(self.channel_names))

    def get_filter(self, index=None):
        """
        Return spatial filter
        :param index:
        :return:
        """
        index = slice(None) if index is None else index
        filter_ = SpatialFilter(self.filters[:, index], self.topographies[:, index])
        return filter_

    def get_filter_sequence(self, index=None):
        return FilterSequence([self.temporal_filter, self.get_filter(index)])

    def get_rejections(self, indexes):
        unmixing_matrix = self.filters.copy()
        inv = np.linalg.pinv(unmixing_matrix)
        unmixing_matrix[:, indexes] = 0
        self.rejection = SpatialRejection(np.dot(unmixing_matrix, inv), rank=len(indexes), type_str=self.name,
                                          topographies=self.topographies[:, indexes])


class CSPDecomposition(SpatialDecomposition):
    def __init__(self, channel_names, fs, band=None, reg_coef=None):
        super(CSPDecomposition, self).__init__(channel_names, fs, band)
        self.reg_coef = reg_coef if (reg_coef is not None) else DEFAULTS['regularizator']
        self.name = 'csp'

    def decompose(self, X, y=None):
        #y = y or np.append(np.zeros((X.shape[0]//2)), np.ones((X.shape[0]//2 + X.shape[0]%2)))
        if y is None:
            raise ValueError('Y is None, but it must includes labels for CSP')
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
        

class CSPDecompositionStimulus(CSPDecomposition):
    def __init__(self, channel_names, fs, band=None, reg_coef=None, pre_interval=500, post_interval=500):
        super(CSPDecompositionStimulus, self).__init__(channel_names, fs, band, reg_coef)
        self.reg_coef = reg_coef if (reg_coef is not None) else DEFAULTS['regularizator']
        self.name = 'csp-s'
        self.pre_interval = pre_interval
        self.post_interval = post_interval
        
    def set_parameters(self, **parameters):
        super(CSPDecomposition, self).set_parameters(**parameters)
        self.reg_coef = parameters['regularizator']
        self.pre_interval = parameters['prestim_interval']
        self.post_interval = parameters['poststim_interval']
        
    def decompose(self, X, y=None):
        y = stimulus_split(y, self.pre_interval, self.post_interval)
        return super(CSPDecompositionStimulus, self).decompose(X[y>=0], y[y>=0])


class ICADecomposition(SpatialDecomposition):
    def __init__(self, channel_names, fs, band=None):
        super(ICADecomposition, self).__init__(channel_names, fs, band)
        self.sorted_channel_index = 0
        self.name = 'ica'

    def decompose(self, X, y=None):
        raw_inst = RawArray(X.T, create_info(self.channel_names, self.fs, 'eeg', None))
        if packaging.version.parse(mne.__version__) >= packaging.version.parse("0.19"):  # validate mne version (mne 0.19+)
            ica = ICA(method='infomax', fit_params=dict(extended=True))
        else:
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
            self.indexes = self.indexes[None, :].repeat(len(self.bands), 0)
        self.pool = [Decomposition(channel_names, fs, band) for band in self.bands]

    def fit(self, X, y=None):
        for decomposer in self.pool:
            decomposer.fit(X, y)

    def get_filter(self):
        filters = [dec.filters[:, index] for dec, index in zip(self.pool, self.indexes)]
        topographies = [dec.topographies[:, index] for dec, index in zip(self.pool, self.indexes)]
        return SpatialFilter(np.hstack(filters), np.hstack(topographies))

    def get_filter_stack(self):
        filters = [dec.get_filter_sequence(index) for dec, index in zip(self.pool, self.indexes)]
        return FilterStack(filters)


class ArtifactRejector:
    def __init__(self, channel_names, fs):
        self.ica = ICADecomposition(channel_names, fs)
        self.rejection = None

    def fit(self, X, y=None):
        scores, filters, topographies = self.ica.decompose(X, y)
        sorted_indexes = np.argsort(scores)[::-1]
        print(np.dot(filters, topographies.T))
        print(np.array(scores)[sorted_indexes])
        filters[:, sorted_indexes[0]] = 0
        self.rejection = SpatialRejection(np.dot(filters, topographies.T))

    def apply(self, chunk: np.ndarray):
        return self.rejection.apply(chunk)

if __name__ == '__main__':
    np.random.seed(42)
    fs = 250
    n_channels = 3
    n_samples = 50001
    t = np.arange(2000)/fs
    #from ..signal_processing.filters import ButterFilter
    import pylab as plt
    alpha = ButterFilter((8, 11), fs, 1).apply(np.random.normal(size=(n_samples, 1)))
    theta = ButterFilter((4, 7), fs, 1).apply(np.random.normal(size=(n_samples, 1)))
    beta = ButterFilter((17, 22), fs, 1).apply(np.random.normal(size=(n_samples, 1)))
    noise = np.random.normal(size=(n_samples, n_channels))*0.05
    alpha_mask = np.ones_like(alpha)
    alpha_mask[n_samples // 2:] *= 0.1
    beta_mask = np.ones_like(beta)
    beta_mask[:n_samples // 2] *= 0.1
    sources = noise
    sources[:, [0]] += alpha * alpha_mask
    sources[:, [1]] += theta
    sources[:, [2]] += beta * beta_mask




    mixing = np.random.randint(-10, 10, size=(3, 3))/10
    unmixing = np.linalg.inv(mixing)
    sensors = np.dot(sources, mixing)

    outliers_indexes = np.random.randint(0, n_samples, 10)
    print(sorted(outliers_indexes))
    sensors[outliers_indexes] *= 10

    labels = np.append(np.zeros((sensors.shape[0]//2)), np.ones((sensors.shape[0]//2 + sensors.shape[0]%2)))
    csp_filter = CSPDecomposition(['Pz', 'Fp1', 'C3'], fs).fit(sensors, labels).get_filter()
    csp_unmixing = csp_filter.filters
    print(np.abs(unmixing/np.abs(unmixing).max(0)) - np.abs(csp_unmixing))


    f, ax = plt.subplots(3, 1)
    ax[0].plot(sources + np.array([1, 0, -1]))
    ax[1].plot(sensors + np.array([1, 0, -1]))
    ax[2].plot(csp_filter.apply(sensors) + np.array([1, 0, -1]))
    plt.show()
