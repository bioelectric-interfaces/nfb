from pynfb.protocols_widgets import *
import numpy as np

class Protocol:
    def __init__(self, signals, source_signal_id=None, name='', duration=30, update_statistics_in_the_end=False,
                 mock_samples_path = (None, None)):
        """ Constructor
        :param signals: derived signals
        :param source_signal_id: base signal id, or None if 'All' signals using
        :param name: name of protocol
        :param duration: duration of protocol
        :param update_statistics_in_the_end: if true update mean and std scaling parameters of signals
        """
        self.update_statistics_in_the_end = update_statistics_in_the_end
        self.mock_samples_file_path, self.mock_samples_protocol = mock_samples_path
        self.name = name
        self.duration = duration
        self.widget_painter = None
        self.signals = signals
        self.source_signal_id = source_signal_id
        pass

    def update_state(self, samples, chunk_size=1):
        if self.source_signal_id is not None:
            self.widget_painter.redraw_state(samples[self.source_signal_id])
        else:
            self.widget_painter.redraw_state(samples[0]) #TODO: if source signal is 'ALL'

    def update_statistics(self):
        pass

    def close_protocol(self):
        if self.update_statistics_in_the_end:
            if self.source_signal_id is not None:
                self.signals[self.source_signal_id].update_statistics()
                self.signals[self.source_signal_id].enable_scaling()
            else:
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