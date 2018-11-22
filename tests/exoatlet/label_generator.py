import numpy as np
from collections import namedtuple

class LabelGenerator:
    def __init__(self, fs, go_duration=8, stay_duration=8, wait_min_duration=2, wait_max_duration=5, max_repeats=3):
        self.wait_n_samples_range = (wait_min_duration * fs, wait_max_duration * fs + 1)
        self.n_samples_dict = {'Go': go_duration*fs, 'Stay': stay_duration*fs, 'Wait': self._generate_wait_n_samples()}
        self.labels_dict = {'Go': 2, 'Stay': 0, 'Wait': 1}
        self.counter = 0
        self.state = 'Stay'
        self.max_repeats = max_repeats
        self.repeats = {'state': 'Stay', 'counter': 1}

    def _generate_wait_n_samples(self):
        return np.random.randint(*self.wait_n_samples_range)

    def apply(self, chunk):
        n_samples = len(chunk)
        if self.counter > self.n_samples_dict[self.state]:
            self.counter = 0
            if self.state in ['Stay', 'Go']:
                if self.repeats['state'] == self.state:
                    self.repeats['counter'] += 1
                else:
                    self.repeats = {'state': self.state, 'counter': 1}
                self.state = 'Wait'
            elif self.state == 'Wait':
                self.n_samples_dict['Wait'] = self._generate_wait_n_samples()
                if self.repeats['counter'] >= self.max_repeats:
                    self.state = ['Stay', 'Go'][0 if self.repeats['state'] == 'Go' else 1]
                else:
                    self.state = ['Stay', 'Go'][np.random.randint(0, 2)]
        self.counter += n_samples
        return np.ones(n_samples)*self.labels_dict[self.state]


if __name__ == '__main__':
    fs = 250
    label_generator = LabelGenerator(fs)
    ts = np.concatenate([label_generator.apply(np.ones(10)) for k in range(20000)])
    import pylab as plt
    plt.plot(np.arange(len(ts))/fs, ts)
    plt.show()