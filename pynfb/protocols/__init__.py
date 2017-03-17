from pynfb.helpers.beep import SingleBeep
from pynfb.protocols.widgets import *
from pynfb.protocols.user_inputs import SelectSSDFilterWidget
from pynfb.widgets.helpers import ch_names_to_2d_pos
from pynfb.widgets.spatial_filter_setup import SpatialFilterSetup
from pynfb.widgets.update_signals_dialog import SignalsSSDManager
from copy import deepcopy
from numpy.random import randint
from numpy import vstack
from PyQt4.QtCore import QCoreApplication
from pynfb.signals import CompositeSignal, DerivedSignal
from pynfb.io.hdf5 import load_h5py


class Protocol:
    def __init__(self, signals, source_signal_id=None, name='', duration=30, update_statistics_in_the_end=False,
                 mock_samples_path=(None, None), show_reward=False, reward_signal_id=0, reward_threshold=0.,
                 ssd_in_the_end=False, timer=None, freq=500, ch_names=None, mock_previous=0, drop_outliers=0,
                 experiment=None, pause_after=False, reverse_mock_previous=False, m_signal_index=None):
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
        self.mock_previous = mock_previous
        self.reverse_mock_previous = reverse_mock_previous
        self.drop_outliers = drop_outliers
        self.experiment = experiment
        self.pause_after = pause_after
        self.m_signal_id = m_signal_index
        pass

    def update_state(self, samples, chunk_size=1, is_half_time=False):
        m_sample = None if self.m_signal_id is None else samples[self.m_signal_id]
        if self.source_signal_id is not None:
            if self.mock_previous == 0:
                self.widget_painter.redraw_state(samples[self.source_signal_id], m_sample)
            else:
                mock_chunk = self.mock_recordings[self.mock_samples_counter:self.mock_samples_counter + chunk_size]
                for signal in self.mock:
                    signal.update(mock_chunk)
                self.mock_samples_counter += chunk_size
                self.mock_samples_counter %= self.mock_recordings.shape[0]
                self.widget_painter.redraw_state(self.mock[self.source_signal_id].current_sample, m_sample)
        else:
            self.widget_painter.redraw_state(samples[0], m_sample)  # if source signal is 'ALL'

    def update_statistics(self):
        pass

    def prepare_raw_mock_if_necessary(self, mock_raw, current_protocol_index):
        print('cpi', current_protocol_index)
        print('mock shape', mock_raw.shape)
        if self.mock_previous:
            if self.source_signal_id is None:
                raise ValueError('If mock_previous is True, source signal should be single')
            self.mock_samples_counter = 0
            self.mock = deepcopy(self.signals)
            for signal in self.mock:
                if isinstance(signal, CompositeSignal):
                    signal.signals = [self.mock[j] for j in range(len(signal.signals))]
            rand_start_ind = randint(0, mock_raw.shape[0])
            self.mock_recordings = vstack((mock_raw[rand_start_ind:], mock_raw[:rand_start_ind]))
            if self.reverse_mock_previous:
                self.mock_recordings = self.mock_recordings[::-1]
            print('**** Success prepare')

    def close_protocol(self, raw=None, signals=None, protocols=list()):
        # action if ssd in the end checkbox was checked
        if self.ssd_in_the_end:

            # stop main timer
            if self.timer:
                self.timer.stop()

            # get spatial filter
            channels_names = self.ch_names
            x = raw
            pos = ch_names_to_2d_pos(channels_names)

            signal_manager = SignalsSSDManager(self.signals, x, pos, channels_names, self, signals, protocols,
                                               sampling_freq=self.freq)
            signal_manager.test_signal.connect(lambda: self.experiment.start_test_protocol(
                protocols[signal_manager.combo_protocols.currentIndex()]
            ))
            signal_manager.test_closed_signal.connect(self.experiment.close_test_protocol)
            signal_manager.exec_()

            # run main timer
            if self.timer:
                self.timer.start(1000 * 1. / self.freq)

        self.update_mean_std(raw, signals)

        if self.pause_after:
            self.experiment.handle_channels_trouble_pause()

    def update_mean_std(self, raw, signals, must=False):
        # update statistics action
        if self.update_statistics_in_the_end or must:
            stats_previous = [(signal.mean, signal.std) for signal in self.signals]
            if self.source_signal_id is not None:
                self.signals[self.source_signal_id].update_statistics(raw=raw, emulate=self.ssd_in_the_end,
                                                                      stats_previous=stats_previous,
                                                                      signals_recorder=signals,
                                                                      drop_outliers=self.drop_outliers)
                self.signals[self.source_signal_id].enable_scaling()
            else:
                updated_derived_signals_recorder = []
                for s, signal in enumerate([signal for signal in self.signals if isinstance(signal, DerivedSignal)]):
                    updated_derived_signals_recorder.append(
                        signal.update_statistics(raw=raw, emulate=self.ssd_in_the_end,
                                                 stats_previous=stats_previous,
                                                 signals_recorder=signals,
                                                 drop_outliers=self.drop_outliers
                                                 ))
                    signal.enable_scaling()
                updated_derived_signals_recorder = np.array(updated_derived_signals_recorder).T
                for signal in [signal for signal in self.signals if isinstance(signal, CompositeSignal)]:
                    signal.update_statistics(raw=raw,
                                             stats_previous=stats_previous,
                                             signals_recorder=signals,
                                             updated_derived_signals_recorder=updated_derived_signals_recorder,
                                             drop_outliers=self.drop_outliers)
                    signal.enable_scaling()


class BaselineProtocol(Protocol):
    def __init__(self, signals, name='Baseline', update_statistics_in_the_end=True, text='Relax', half_time_text=None,
                 **kwargs):
        kwargs['name'] = name
        kwargs['update_statistics_in_the_end'] = update_statistics_in_the_end
        super().__init__(signals, **kwargs)
        self.text = text
        self.widget_painter = BaselineProtocolWidgetPainter(text=text, show_reward=self.show_reward)
        self.half_time_text_change = half_time_text is not None
        self.half_time_text = half_time_text
        self.is_half_time = False
        self.beep = SingleBeep()
        pass

    def update_state(self, samples, chunk_size=1, is_half_time=False):
        if self.half_time_text_change:
            if is_half_time and not self.is_half_time:
                self.beep.try_to_play()
                self.is_half_time = True
                self.widget_painter.set_message(self.half_time_text)

    def close_protocol(self, **kwargs):
        self.is_half_time = False
        self.beep = SingleBeep()
        self.widget_painter.set_message('')
        super(BaselineProtocol, self).close_protocol(**kwargs)
        self.widget_painter.set_message(self.text)


class FeedbackProtocol(Protocol):
    def __init__(self, signals, name='Feedback', circle_border=0, m_threshold=1, **kwargs):
        kwargs['name'] = name
        super().__init__(signals, **kwargs)
        self.widget_painter = CircleFeedbackProtocolWidgetPainter(show_reward=self.show_reward,
                                                                  circle_border=circle_border,
                                                                  m_threshold=m_threshold)
        pass


class ThresholdBlinkFeedbackProtocol(Protocol):
    def __init__(self, signals, name='ThresholdBlink', threshold=1000, time_ms=50, **kwargs):
        kwargs['name'] = name
        super().__init__(signals, **kwargs)
        self.widget_painter = ThresholdBlinkFeedbackProtocolWidgetPainter(threshold=threshold, time_ms=time_ms,
                                                                          show_reward=self.show_reward)

class VideoProtocol(Protocol):
    def __init__(self, signals, name='Video', video_path='', **kwargs):
        kwargs['name'] = name
        super().__init__(signals, **kwargs)
        self.widget_painter = VideoProtocolWidgetPainter(video_file_path=video_path)
        pass

class SSDProtocol(Protocol):
    def __init__(self, signals, text='Relax', **kwargs):
        kwargs['ssd_in_the_end'] = True
        super().__init__(signals, **kwargs)
        self.widget_painter = BaselineProtocolWidgetPainter(text=text, show_reward=self.show_reward)


def main():
    pass


if __name__ == '__main__':
    main()
