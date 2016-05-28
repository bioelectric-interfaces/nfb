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
    def __init__(self, signals, name='Baseline', update_statistics_in_the_end=True, **kwargs):
        kwargs['name'] = name
        kwargs['update_statistics_in_the_end'] = update_statistics_in_the_end
        super().__init__(signals, **kwargs)
        self.widget_painter = BaselineProtocolWidgetPainter()
        pass




class FeedbackProtocol(Protocol):
    def __init__(self, signals, name='Feedback', **kwargs):
        kwargs['name'] = name
        super().__init__(signals, **kwargs)
        self.widget_painter = CircleFeedbackProtocolWidgetPainter()
        pass


class ThresholdBlinkFeedbackProtocol(Protocol):
    def __init__(self, signals, name='ThresholdBlink', threshold=1000, time_ms=50, **kwargs):
        kwargs['name'] = name
        super().__init__(signals, **kwargs)
        self.widget_painter = ThresholdBlinkFeedbackProtocolWidgetPainter(threshold=threshold, time_ms=time_ms)

def main():
    pass


if __name__ == '__main__':
    main()