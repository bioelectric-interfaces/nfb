import numpy as np
import sympy

class CompositeSignal:
    """
    Class for composite signal
    """
    def __init__(self, signals, expression, name):
        """
        Constructor
        :param signals: list of all signals
        :param expression: str expression
        """
        self.name = name
        self.signals = signals
        self._signals_names = [signal.name for signal in self.signals]
        self.expression = sympy.sympify(expression if expression != '' else '0')
        self.expression_lambda = sympy.lambdify(self._signals_names, self.expression, modules="numpy")
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
        self.current_sample = self.expression_lambda(*[signal.current_sample for signal in self.signals])
        self.mean_acc = (self.n_acc * self.mean_acc + chunk_size * self.current_sample) / (self.n_acc + chunk_size)
        self.var_acc = (self.n_acc * self.var_acc + chunk_size * (self.current_sample - self.mean_acc) ** 2) / (
            self.n_acc + chunk_size)
        self.std_acc = self.var_acc ** 0.5
        self.n_acc += chunk_size
        if self.scaling_flag and self.std>0:
            self.current_sample = (self.current_sample - self.mean) / self.std
        pass

    def update_statistics(self, mean=None, std=None, raw=None, emulate=False,
                          signals_recorder=None, stats_previous=None):
        from time import time
        timer = time()
        signals_data = signals_recorder[:, :len(self.signals)].copy()
        for j, signal_data in enumerate(signals_data.T):
            if np.isfinite(stats_previous[j][0]) and np.isfinite(stats_previous[j][1]):
                signal_data = signal_data * stats_previous[j][1] + stats_previous[j][0]
            signals_data[:, j] = (signal_data - self.signals[j].mean) / self.signals[j].std
        actual_data = self.expression_lambda(*signals_data.T)
        self.mean = mean if (mean is not None) else actual_data.mean()
        self.std = std if (std is not None) else actual_data.std()
        self.reset_statistic_acc()
        print('*** COMPOSITE TIME ELAPSED', time() - timer)

    def reset_statistic_acc(self):
        self.mean_acc = 0
        self.var_acc = 0
        self.std_acc = 0
        self.n_acc = 0

    def enable_scaling(self):
        self.scaling_flag = True
        pass