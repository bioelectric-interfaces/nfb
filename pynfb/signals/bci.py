from ..signal_processing.filters import ButterFilter, FilterSequence, FilterStack, InstantaneousVarianceFilter
from ..signal_processing.decompositions import SpatialDecompositionPool
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from ..signal_processing.helpers import get_outliers_mask
#from ..signals._bci_dev.bcimodel_draft import BCIModel
BANDS_DEFAULT = [(6, 10), (8, 12), (10, 14), (12, 16), (14, 18), (16, 20), (18, 22), (20, 24)]
STATES_LABELS_DEFAULT = [0, 1, 2]
INDEXES_DEFAULT = [1, -1]
from collections import Counter
from sklearn.preprocessing import StandardScaler


class BCIModel():
    def __init__(self, fs, bands, ch_names, states_labels, indexes):
        self.states_labels = states_labels
        self.bands = bands
        self.prefilter = FilterSequence([ButterFilter((0.5, 45), fs, len(ch_names))])
        self.csp_pools = [SpatialDecompositionPool(ch_names, fs, bands, 'csp', indexes) for _label in states_labels]
        self.csp_transformer = None
        self.var_detector = InstantaneousVarianceFilter(len(bands)*len(indexes)*len(states_labels), n_taps=int(fs//2))
        #self.classifier = MLPClassifier(hidden_layer_sizes=(), early_stopping=True, verbose=True)
        self.classifier = RandomForestClassifier(max_depth=3, min_samples_leaf=100)
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X = self.prefilter.apply(X)
        for csp_pool, label in zip(self.csp_pools, self.states_labels):
            csp_pool.fit(X, y == label)
        self.csp_transformer = FilterStack([pool.get_filter_stack() for pool in self.csp_pools])
        X = self.csp_transformer.apply(X)
        X = self.var_detector.apply(X)
        X = self.scaler.fit_transform(X)
        self.classifier.fit(X, y)
        accuracies = [sum(self.classifier.predict(X) == y)/len(y)]
        print('Fit accuracy {}'.format(accuracies[0]))
        for label in self.states_labels:
            accuracies.append(sum(self.classifier.predict(X[y == label]) == label) / sum(y == label))
            print('Fit accuracy label {}: {}'.format(label,accuracies[-1]))
        return accuracies

    def get_accuracies(self, X, y):
        accuracies = [sum(self.apply(X) == y) / len(y)]
        for label in self.states_labels:
            accuracies.append(sum(self.apply(X[y == label]) == label) / sum(y == label))
        return accuracies

    def apply(self, chunk: np.ndarray):
        chunk = self.prefilter.apply(chunk)
        chunk = self.csp_transformer.apply(chunk)
        chunk = self.var_detector.apply(chunk)
        chunk = self.scaler.transform(chunk)
        predicted_labels = self.classifier.predict(chunk)
        return predicted_labels

class BCISignal():
    def __init__(self, fs, ch_names, name, id, bands=None, states_labels=None, indexes=None):
        bands = bands if bands is not None else BANDS_DEFAULT
        states_labels = states_labels if states_labels is not None else STATES_LABELS_DEFAULT
        indexes = indexes if indexes is not None else INDEXES_DEFAULT
        self.model_args = [fs, bands, ch_names, states_labels, indexes]
        self.model = BCIModel(*self.model_args)
        self.current_sample = 0
        self.name = name
        self.id = id
        self.mean = 0
        self.std = 0
        self.scaling_flag = False
        self.model_fitted = False
        self.current_chunk = None

    def update(self, chunk):
        if self.model_fitted:
            labels = self.model.apply(chunk)
            self.current_sample = Counter(labels).most_common(1)[0][0]
        self.current_chunk = self.current_sample * np.ones(len(chunk))
        with open("bci_current_state.pkl", "w", encoding="utf-8") as fp:
            fp.write(str(self.current_sample))
        #print(self.current_sample, type(self.current_sample))

    def apply(self, chunk):
        return self.model.apply(chunk)

    def fit_model(self, X, y):
        accuracies = self.model.fit(X, y)
        self.model_fitted = True
        return accuracies

    def reset_model(self):
        self.model = BCIModel(*self.model_args)

    def reset_statistic_acc(self):
        pass

    def enable_scaling(self):
        pass

    def descale_recording(self, data):
        return data