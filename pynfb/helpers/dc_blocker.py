import numpy as np


class DCBlocker:
    def __init__(self, r=0.99):
        self.last_y = 0
        self.r = r

    def filter(self, x, r=0.99):
        # DC Blocker https://ccrma.stanford.edu/~jos/fp/DC_Blocker.html
        y = np.zeros_like(x)
        y[0] = self.last_y
        for n in range(1, x.shape[0]):
            y[n] = x[n] - x[n - 1] + r * y[n - 1]
        self.last_y = y[-1]
        return y

    def apply(self, x: np.ndarray):
        # DC Blocker https://ccrma.stanford.edu/~jos/fp/DC_Blocker.html
        y = np.zeros_like(x)
        y[0] = self.last_y
        for n in range(1, x.shape[0]):
            y[n] = x[n] - x[n - 1] + self.r * y[n - 1]
        self.last_y = y[-1]
        return y