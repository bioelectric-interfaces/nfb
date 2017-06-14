from pynfb.helpers.dc_blocker import DCBlocker
from pynfb.signal_processing.filters import ButterFilter, FilterSequence, FilterStack, InstantaneousVarianceFilter
from pynfb.signal_processing.decompositions import SpatialDecompositionPool
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from pynfb.signal_processing.helpers import get_outliers_mask
from pynfb.signals._bci_dev.bcimodel_draft import BCIModel


class BCISignal():
    def __init__(self, fs, bands, ch_names, states_labels, indexes):
        self.states_labels = states_labels
        self.bands = bands
        self.prefilter = FilterSequence([ButterFilter((0.5, 45), fs, len(ch_names))])
        self.csp_pools = [SpatialDecompositionPool(ch_names, fs, bands, 'csp', indexes) for _label in states_labels]
        self.csp_transformer = None
        self.var_detector = InstantaneousVarianceFilter(len(bands)*len(indexes)*len(states_labels), n_taps=fs//2)
        self.classifier = MLPClassifier(hidden_layer_sizes=(), early_stopping=True, verbose=True)
        #self.classifier = RandomForestClassifier(max_depth=3, min_samples_leaf=100)

    def fit(self, X, y=None):
        X = self.prefilter.apply(X)
        for csp_pool, label in zip(self.csp_pools, self.states_labels):
            csp_pool.fit(X, y == label)
        self.csp_transformer = FilterStack([pool.get_filter_stack() for pool in self.csp_pools])
        X = self.csp_transformer.apply(X)
        X = self.var_detector.apply(X)

        ##outliers_mask = get_outliers_mask(X)
        #good_mask = ~get_outliers_mask(X, iter_numb=5, std=4)
        #X=X[good_mask]
        #y=y[good_mask]
        #X = X[1000:]
        #y = y[1000:]
        import pylab as plt
        plt.plot(X + np.arange(len(X[0])))
        plt.show()
        self.classifier.fit(X, y)
        print('Fit accuracy {}'.format(sum(self.classifier.predict(X) == y)/len(y)))

    def apply(self, chunk: np.ndarray):
        chunk = self.prefilter.apply(chunk)
        chunk = self.csp_transformer.apply(chunk)
        chunk = self.var_detector.apply(chunk)
        predicted_labels = self.classifier.predict(chunk)
        return predicted_labels

if __name__ == '__main__':
    np.random.seed(42)

    # loading anp pre processing
    filenames = ['C:\\Users\\nsmetanin\\PycharmProjects\\nfb\\pynfb\\signals\\_bci_dev\\sm_ksenia_1.mat',
                'C:\\Users\\nsmetanin\\PycharmProjects\\nfb\\pynfb\\signals\\_bci_dev\\sm_ksenia_2.mat']
    datasets = []
    for filename in filenames:
        [eeg_data, states_labels1, fs, chan_names, chan_numb, samp_numb, states_codes] = BCIModel.open_eeg_mat(filename,
                                                                                                        centered=False)

        fs = fs[0, 0]
        nozeros_mask = np.sum(eeg_data[:, :fs * 2], 1) != 0  # Detect constant (zero) channels
        without_emp_mask = nozeros_mask & (chan_names[0, :] != 'A1') & (chan_names[0, :] != 'A2') & (chan_names[0, :] != 'AUX')
        eeg_data = eeg_data[without_emp_mask, :].T  # Remove constant (zero) channels and prespecified channels
        ch_names = [name[0][0] for name in chan_names[:, without_emp_mask].T]
        labels = states_labels1[0]
        datasets.append((eeg_data, labels))

    # fit model
    bands = [(6, 10), (8, 12), (10, 14), (12, 16), (14, 18), (16, 20), (18, 22), (20, 24)]
    state_labels = [1, 2, 6]
    indexes = [-1, 1]
    bci_signal = BCISignal(fs, bands, ch_names, state_labels, indexes)
    bci_signal.fit(*datasets[0])

    # test model
    X, y = datasets[1]
    print('Test file accuracy {}'.format(sum(bci_signal.apply(X) == y) / len(y)))

    chunk_size = 8
    chuncked_y = np.hstack([bci_signal.apply(X[k * chunk_size:(k + 1) * chunk_size]) for k in range(len(X) // chunk_size)])
    print('Chunked test accuracy {}'.format(sum(chuncked_y == y[:len(chuncked_y)]) / len(chuncked_y)))