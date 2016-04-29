from pynfb.protocols_widgets import *
import numpy as np

class Protocol:
    def __init__(self, name='', duration=30):
        self.name = name
        self.duration = duration
        self.widget = ProtocolWidget()
        pass

    def update_state(self, sample, chunk_size=1):
        self.widget.redraw_state(sample)


class BaselineProtocol(Protocol):
    def __init__(self, duration=30):
        super().__init__(name='Baseline', duration=duration)
        self.widget = BaselineProtocolWidget()
        self.mean = 0
        self.var = 0
        self.std  = 0
        self.n = 0
        pass

    def update_state(self, sample, chunk_size=1):
        self.mean = (self.n*self.mean + chunk_size*sample)/(self.n+chunk_size)
        self.var = (self.n*self.var + chunk_size*(sample - self.mean)**2)/(self.n+chunk_size)
        self.std = self.var**0.5
        self.n += chunk_size
        self.widget.redraw_state((self.mean, self.std)) # TODO: delete
        #print(np.mean(self.accumulator), np.std(self.accumulator))

class FeedbackProtocol(Protocol):
    def __init__(self, duration=30):
        super().__init__(name='Feedback', duration=duration)
        self.widget = CircleFeedbackProtocolWidget()
        pass


def main():
    p = FeedbackProtocol()


if __name__ == '__main__':
    main()