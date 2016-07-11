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
        current_samples = [(signal.name, signal.current_sample) for signal in self.signals]
        self.current_sample = float(self.expression.subs(current_samples))
        self.mean_acc = (self.n_acc * self.mean_acc + chunk_size * self.current_sample) / (self.n_acc + chunk_size)
        self.var_acc = (self.n_acc * self.var_acc + chunk_size * (self.current_sample - self.mean_acc) ** 2) / (
            self.n_acc + chunk_size)
        self.std_acc = self.var_acc ** 0.5
        self.n_acc += chunk_size
        if self.scaling_flag:
            self.current_sample = (self.current_sample - self.mean) / self.std
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