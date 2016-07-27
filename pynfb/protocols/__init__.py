from pynfb.protocols.widgets import *
from pynfb.protocols.user_inputs import SelectSSDFilterWidget
from pynfb.widgets.helpers import ch_names_to_2d_pos
from pynfb.widgets.spatial_filter_setup import SpatialFilterSetup
from pynfb.widgets.update_signals_dialog import SignalsSSDManager


class Protocol:
    def __init__(self, signals, source_signal_id=None, name='', duration=30, update_statistics_in_the_end=False,
                 mock_samples_path=(None, None), show_reward=False, reward_signal_id=0, reward_threshold=0.,
                 ssd_in_the_end=False, timer=None, freq=500, ch_names=None):
        """ Constructor
        :param signals: derived signals
        :param source_signal_id: base signal id, or None if 'All' signals using
        :param name: name of protocol
        :param duration: duration of protocol
        :param update_statistics_in_the_end: if true update mean and std scaling parameters of signals
        """
        self.show_reward = show_reward
        self.reward_signal_id = reward_signal_id
        self.reward_threshold = reward_threshold
        self.update_statistics_in_the_end = update_statistics_in_the_end
        self.mock_samples_file_path, self.mock_samples_protocol = mock_samples_path
        self.name = name
        self.duration = duration
        self.widget_painter = None
        self.signals = signals
        self.source_signal_id = source_signal_id
        self.ssd_in_the_end = ssd_in_the_end
        self.timer = timer
        self.freq = freq
        self.ch_names = ch_names
        pass

    def update_state(self, samples, chunk_size=1):
        if self.source_signal_id is not None:
            self.widget_painter.redraw_state(samples[self.source_signal_id])
        else:
            self.widget_painter.redraw_state(samples[0])  # if source signal is 'ALL'

    def update_statistics(self):
        pass

    def close_protocol(self, raw=None, signals=None):
        # action if ssd in the end checkbox was checked
        if self.ssd_in_the_end:

            # stop main timer
            if self.timer:
                self.timer.stop()

            # get spatial filter
            channels_names = self.ch_names
            x = raw
            pos = ch_names_to_2d_pos(channels_names)

            signal_manager = SignalsSSDManager(self.signals, x, pos, channels_names, self.freq)
            if self.source_signal_id is not None:
                signal_manager.run_ssd(row=self.source_signal_id)
            else:
                signal_manager.exec_()

            # run main timer
            if self.timer:
                self.timer.start(1000 * 1. / self.freq)

        # update statistics action
        if self.update_statistics_in_the_end:
            if self.source_signal_id is not None:
                self.signals[self.source_signal_id].update_statistics(raw=raw, emulate=self.ssd_in_the_end)
                self.signals[self.source_signal_id].enable_scaling()
            else:
                for signal in self.signals:
                    signal.update_statistics(raw=raw, emulate=self.ssd_in_the_end)
                    signal.enable_scaling()


class BaselineProtocol(Protocol):
    def __init__(self, signals, name='Baseline', update_statistics_in_the_end=True, text='Relax', **kwargs):
        kwargs['name'] = name
        kwargs['update_statistics_in_the_end'] = update_statistics_in_the_end
        super().__init__(signals, **kwargs)
        self.widget_painter = BaselineProtocolWidgetPainter(text=text, show_reward=self.show_reward)
        pass


class FeedbackProtocol(Protocol):
    def __init__(self, signals, name='Feedback', **kwargs):
        kwargs['name'] = name
        super().__init__(signals, **kwargs)
        self.widget_painter = CircleFeedbackProtocolWidgetPainter(show_reward=self.show_reward)
        pass


class ThresholdBlinkFeedbackProtocol(Protocol):
    def __init__(self, signals, name='ThresholdBlink', threshold=1000, time_ms=50, **kwargs):
        kwargs['name'] = name
        super().__init__(signals, **kwargs)
        self.widget_painter = ThresholdBlinkFeedbackProtocolWidgetPainter(threshold=threshold, time_ms=time_ms,
                                                                          show_reward=self.show_reward)


class SSDProtocol(Protocol):
    def __init__(self, signals, text='Relax', **kwargs):
        kwargs['ssd_in_the_end'] = True
        super().__init__(signals, **kwargs)
        self.widget_painter = BaselineProtocolWidgetPainter(text=text, show_reward=self.show_reward)


def main():
    pass


if __name__ == '__main__':
    main()
