from pynfb.helpers.dc_blocker import DCBlocker
from pynfb.postprocessing.utils import get_info
from pynfb.signal_processing.filters import ButterFilter, FilterSequence, FilterStack, InstantaneousVarianceFilter
from pynfb.signal_processing.decompositions import SpatialDecompositionPool
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from pynfb.signal_processing.helpers import get_outliers_mask
from pynfb.signals._bci_dev.bcimodel_draft import BCIModel
from pynfb.signals.bci import BCISignal


class BCISignal1():
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

    import h5py
    import pylab as plt

    file = r'C:\Users\Nikolai\PycharmProjects\nfb\pynfb\results\BCI_Test_4_06-23_16-20-48\experiment_data.h5'
    #file = r'C:\Users\Nikolai\PycharmProjects\nfb\pynfb\results\BCI_Test_2_1_06-22_17-01-07\experiment_data.h5'

    labels_map = {'Open': 0, 'Right': 2, 'Left': 1}
    with h5py.File(file) as f:
        fs, ch_names, p_names = get_info(f, []) #['AUX', 'A1', 'A2', 'F4', 'Pz']
        before = []
        after = []
        before_labels = []
        after_labels = []
        k = 0
        for protocol in ['protocol{}'.format(j + 1) for j in range(len(f.keys()) - 3)]:
            name = f[protocol].attrs['name']
            if name in ['Right', 'Open', 'Left']:
                data = f[protocol + '/raw_data'][:]
                labels = np.ones(len(data), dtype=int) * labels_map[name]
                if k < 9:
                    before.append(data)
                    before_labels.append(labels)
                else:
                    after.append(data)
                    after_labels.append(labels)

                k += 1
    stds = 1#np.vstack(before).std(0)

    datasets = [(np.vstack(before)/stds, np.concatenate(before_labels, 0)),
                (np.vstack(after)/stds, np.concatenate(after_labels, 0))][::-1]
    print(datasets[0][0].shape, datasets[1][0].shape)

    # fit model
    bands = [(6, 10), (8, 12), (10, 14), (12, 16), (14, 18), (16, 20), (18, 22), (20, 24)]
    state_labels = [0, 1, 2]
    indexes = [-1, 1]
    bci_signal = BCISignal(fs, ch_names, 'bci', 1, bands,  state_labels, indexes)
    bci_signal.fit_model(*datasets[0])

    # test model
    X, y = datasets[1]
    print('Test file accuracy {}'.format(sum(bci_signal.apply(X) == y) / len(y)))

    chunk_size = 8
    chuncked_y = np.hstack([bci_signal.apply(X[k * chunk_size:(k + 1) * chunk_size]) for k in range(len(X) // chunk_size)])
    print('Chunked test accuracy {}'.format(sum(chuncked_y == y[:len(chuncked_y)]) / len(chuncked_y)))