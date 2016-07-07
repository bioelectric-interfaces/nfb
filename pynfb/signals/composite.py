import numpy as np

class CompositeSignal:
    """
    Class for composite signal
    """
    def __init__(self, signals, weights, operation):
        """
        Constructor
        :param signals: list of signals
        :param weights: list of signal weights
        :param operation: operation type (str)
        Let w_j and s_j are weight and current sample of j signal in list of signals
        If operation == 'sum' composite signal is sum of w_j*s_j
        If operation == 'prod' composite signal is product of s_j^w_j
        """
        self.signals = signals
        self.weights = weights
        self.operation = operation
        self.current_sample = 0
        # signal statistics
        self.scaling_flag = False
        self.mean = np.nan
        self.std = np.nan
        # signal statistics accumulators
        self.mean_acc = 0
        self.var_acc = 0
        self.std_acc = 0
        self.n_acc = 0

    def update(self, chunk):
        chunk_size = chunk.shape[0]
        weights_signals = zip(self.weights, self.signals)
        if self.operation == 'sum':
            self.current_sample = np.sum([signal.current_sample*w for w, signal in weights_signals])
        elif self.operation == 'prod':
            self.current_sample = np.prod([np.power(signal.current_sample, w) for w, signal in weights_signals])
        else:
            raise TypeError('Wrong operation type')
        self.mean_acc = (self.n_acc * self.mean_acc + chunk_size * self.current_sample) / (self.n_acc + chunk_size)
        self.var_acc = (self.n_acc * self.var_acc + chunk_size * (self.current_sample - self.mean_acc) ** 2) / (
            self.n_acc + chunk_size)
        self.std_acc = self.var_acc ** 0.5
        self.n_acc += chunk_size
        pass

    def update_statistics(self, mean=None, std=None):
        self.mean = mean if (mean is not None) else self.mean_acc
        self.std = std if (std is not None) else self.std_acc
        self.reset_statistic_acc()

    def reset_statistic_acc(self):
        self.mean_acc = 0
        self.var_acc = 0
        self.std_acc = 0
        self.n_acc = 0

    def enable_scaling(self):
        self.scaling_flag = True
        pass