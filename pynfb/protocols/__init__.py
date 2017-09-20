from copy import deepcopy

import numpy as np
from numpy import vstack
from numpy.random import randint

from ..helpers.beep import SingleBeep
from ..io.hdf5 import load_h5py_protocols_raw
from ..protocols.user_inputs import SelectSSDFilterWidget
from ..protocols.widgets import (CircleFeedbackProtocolWidgetPainter, BarFeedbackProtocolWidgetPainter,
                                     PsyProtocolWidgetPainter, BaselineProtocolWidgetPainter,
                                     ThresholdBlinkFeedbackProtocolWidgetPainter, VideoProtocolWidgetPainter)
from ..signals import CompositeSignal, DerivedSignal
from ..widgets.helpers import ch_names_to_2d_pos
from ..widgets.update_signals_dialog import SignalsSSDManager


class Protocol:
    def __init__(self, signals, source_signal_id=None, name='', duration=30, update_statistics_in_the_end=False,
                 mock_samples_path=(None, None), show_reward=False, reward_signal_id=0, reward_threshold=0.,
                 ssd_in_the_end=False, timer=None, freq=500, ch_names=None, mock_previous=0, drop_outliers=0,
                 experiment=None, pause_after=False, reverse_mock_previous=False, m_signal_index=None,
                 shuffle_mock_previous=None, beep_after=False, as_mock=False, fast_bci_fit=False):
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
        self.shuffle_mock_previous = shuffle_mock_previous
        self.beep_after = beep_after
        self.as_mock = as_mock
        self.fast_bci_fit = fast_bci_fit
        pass

    def update_state(self, samples, reward, chunk_size=1, is_half_time=False):

        m_sample = None if self.m_signal_id is None else samples[self.m_signal_id]
        if self.source_signal_id is not None:
            if self.mock_previous == 0:
                mark = self.widget_painter.redraw_state(samples[self.source_signal_id], m_sample)
                reward.update(samples[reward.signal_ind], chunk_size)
            else:
                mock_chunk = self.mock_recordings[self.mock_samples_counter:self.mock_samples_counter + chunk_size]
                for signal in self.mock:
                    signal.update(mock_chunk)
                self.mock_samples_counter += chunk_size
                self.mock_samples_counter %= self.mock_recordings.shape[0]
                #mock_signals = self.mock_recordings_signals[self.mock_samples_counter - 1]
                #mark = self.widget_painter.redraw_state(mock_signals[self.source_signal_id], m_sample)
                #reward.update(mock_signals[reward.signal_ind], chunk_size)
                mark = self.widget_painter.redraw_state(self.mock[self.source_signal_id].current_sample, m_sample)
                reward.update(self.mock[reward.signal_ind].current_sample, chunk_size)
        else:
            mark = self.widget_painter.redraw_state(samples[0], m_sample)  # if source signal is 'ALL'
        return mark

    def update_statistics(self):
        pass

    def prepare_raw_mock_if_necessary(self, mock_raw, random_previous_fb_protocol_number, mock_signals):
        if self.shuffle_mock_previous:
            self.mock_previous = random_previous_fb_protocol_number
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
            self.mock_recordings_signals = vstack((mock_signals[rand_start_ind:], mock_signals[:rand_start_ind]))
            if self.reverse_mock_previous:
                self.mock_recordings = self.mock_recordings[::-1]
                self.mock_recordings_signals = self.mock_recordings_signals[::-1]

    def close_protocol(self, raw=None, signals=None, protocols=list(), protocols_seq=None, raw_file=None, marks=None):
        # action if ssd in the end checkbox was checked
        if self.beep_after:
            SingleBeep().try_to_play()

        if self.ssd_in_the_end or self.fast_bci_fit:

            # stop main timer
            if self.timer:
                self.timer.stop()

            # get spatial filter
            channels_names = self.ch_names
            if raw_file is not None and protocols_seq is not None:
                x = load_h5py_protocols_raw(raw_file, [j for j in range(len(protocols_seq)-1)])
                x.append(raw)
            else:
                raise AttributeError('Attributes protocol_seq and raw_file should be not a None')
            pos = ch_names_to_2d_pos(channels_names)

        if self.fast_bci_fit:
            X = [x for x, name in zip(x, protocols_seq) if name in ['Open', 'Left', 'Right']]
            y = [np.ones(len(x), dtype=int) * {'Open': 0, 'Left': 1, 'Right': 2}[name] for x, name in zip(x, protocols_seq)
                 if name in ['Open', 'Left', 'Right']]
            X = np.vstack(X)
            y = np.concatenate(y, 0)
            self.signals[0].fit_model(X, y)

        if self.ssd_in_the_end:
            signal_manager = SignalsSSDManager(self.signals, x, pos, channels_names, self, signals, protocols,
                                               sampling_freq=self.freq, protocol_seq=protocols_seq, marks=marks)
            signal_manager.test_signal.connect(lambda: self.experiment.start_test_protocol(
                protocols[signal_manager.combo_protocols.currentIndex()]
            ))
            signal_manager.test_closed_signal.connect(self.experiment.close_test_protocol)
            signal_manager.exec_()

        if self.ssd_in_the_end or self.fast_bci_fit:
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
                self.signals[self.source_signal_id].update_statistics(from_acc=True)
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

    def update_state(self, samples, reward, chunk_size=1, is_half_time=False):
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
        if circle_border == 2:
            self.widget_painter = BarFeedbackProtocolWidgetPainter(show_reward=self.show_reward,
                                                                      circle_border=circle_border,
                                                                      m_threshold=m_threshold)
        else:
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


class PsyProtocol(Protocol):
    def __init__(self, signals, detection, name='Psy', **kwargs):
        kwargs['name'] = name
        super().__init__(signals, **kwargs)
        self.widget_painter = PsyProtocolWidgetPainter(detection)
        pass

    def close_protocol(self, **kwargs):
        self.widget_painter.close()
        super(PsyProtocol, self).close_protocol(**kwargs)


def main():
    pass


if __name__ == '__main__':
    main()
