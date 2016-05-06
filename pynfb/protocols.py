from pynfb.protocols_widgets import *
import numpy as np

class Protocol:
    def __init__(self, signals, base_signal_id = 0, name='', duration=30, update_statistics_in_the_end=False):
        self.update_statistics_in_the_end = update_statistics_in_the_end
        self.name = name
        self.duration = duration
        self.widget_painter = None
        self.signals = signals
        self.base_signal_id = base_signal_id
        pass

    def update_state(self, samples, chunk_size=1):
        self.widget_painter.redraw_state(samples[self.base_signal_id])

    def update_statistics(self):
        pass

    def close_protocol(self):
        if self.update_statistics_in_the_end:
            for signal in self.signals:
                signal.update_statistics()
                signal.enable_scaling()

class BaselineProtocol(Protocol):
    def __init__(self, signals, base_signal_id = 0, name='Baseline', duration=30, update_statistics_in_the_end=True):
        super().__init__(signals, base_signal_id=base_signal_id, name=name, duration=duration,
                         update_statistics_in_the_end=update_statistics_in_the_end)
        self.widget_painter = BaselineProtocolWidgetPainter()
        pass

    def update_state(self, sample, chunk_size=1):
        self.widget_painter.redraw_state((self.signals[self.base_signal_id].mean_acc,
                                          self.signals[self.base_signal_id].std_acc))




class FeedbackProtocol(Protocol):
    def __init__(self, signals, base_signal_id = 0, name='Feedback', duration=30, update_statistics_in_the_end=False):
        super().__init__(signals, base_signal_id=base_signal_id, name=name, duration=duration,
                         update_statistics_in_the_end=update_statistics_in_the_end)
        self.widget_painter = CircleFeedbackProtocolWidgetPainter()
        pass


def main():
    pass


if __name__ == '__main__':
    main()