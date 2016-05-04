from pynfb.protocols_widgets import *
import numpy as np

class Protocol:
    def __init__(self, signals, name='', duration=30):
        self.name = name
        self.duration = duration
        self.widget = ProtocolWidget()
        self.signals = signals
        pass

    def update_state(self, sample, chunk_size=1):
        self.widget.redraw_state(sample)

    def update_statistics(self):
        pass

class BaselineProtocol(Protocol):
    def __init__(self, signals, duration=30):
        super().__init__(signals = signals, name='Baseline', duration=duration)
        self.widget = BaselineProtocolWidget()
        pass

    def update_state(self, sample, chunk_size=1):
        self.widget.redraw_state((self.signals[0].mean_acc, self.signals[0].std_acc))

class FeedbackProtocol(Protocol):
    def __init__(self, signals, duration=30):
        super().__init__(signals = signals, name='Feedback', duration=duration)
        self.widget = CircleFeedbackProtocolWidget()
        pass


def main():
    pass


if __name__ == '__main__':
    main()