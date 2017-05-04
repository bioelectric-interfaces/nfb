import numpy as np

class BCIModel():
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def fit(self, data):
        pass

class BCISignal():
    def __init__(self, arg1):
        self.model = BCIModel(n_samples=42)
        self.current_state = 0

    def update(self, chunk):
        self.current_state = self.model.n_samples**2

    def fit_model(self, data):
        self.model.fit(data)