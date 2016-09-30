import os
from datetime import datetime
from multiprocessing import Process, Pool
import numpy as np
from PyQt4 import QtCore

from pynfb.widgets.channel_trouble import ChannelTroubleWarning
from pynfb.widgets.helpers import WaitMessage
from .generators import run_eeg_sim
from .inlets.ftbuffer_inlet import FieldTripBufferInlet
from .inlets.lsl_inlet import LSLInlet
from .inlets.channels_selector import ChannelsSelector
from .io.hdf5 import load_h5py_all_samples, save_h5py, load_h5py, save_signals, load_h5py_protocol_signals
from .io.xml import params_to_xml_file
from .io import read_spatial_filter
from .protocols import BaselineProtocol, FeedbackProtocol, ThresholdBlinkFeedbackProtocol, SSDProtocol
from .signals import DerivedSignal, CompositeSignal
from .windows import MainWindow


# helpers
def int_or_none(string):
    return int(string) if len(string) > 0 else None


class Experiment():
    def __init__(self, app, params):
        self.app = app
        self.params = params
        self.main_timer = None
        self.stream = None
        self.thread = None
        self.catch_channels_trouble = True
        timestamp_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
        self.dir_name = 'results/{}_{}/'.format(self.params['sExperimentName'], timestamp_str)
        os.makedirs(self.dir_name)
        self.mock_signals_buffer = None
        self.restart()
        pass

    def update(self):
        """
        Experiment main update action
        :return: None
        """
        # get next chunk
        chunk, other_chunk = self.stream.get_next_chunk() if self.stream is not None else (None, None)
        if chunk is not None:

            # update and collect current samples
            for i, signal in enumerate(self.signals):
                signal.update(chunk)
                self.current_samples[i] = signal.current_sample

            # record data
            if self.main.player_panel.start.isChecked():
                self.subject.figure.update_reward(self.reward.get_score())
                if self.samples_counter < self.experiment_n_samples:
                    chunk_slice = slice(self.samples_counter, self.samples_counter + chunk.shape[0])
                    self.raw_recorder[chunk_slice] = chunk[:, :self.n_channels]
                    self.raw_recorder_other[chunk_slice] = other_chunk
                    for s, sample in enumerate(self.current_samples):
                        self.signals_recorder[chunk_slice, s] = sample
                    self.samples_counter += chunk.shape[0]

                    # catch channels trouble
                    if self.samples_counter > self.seconds:
                        self.seconds += 2 * self.freq
                        raw_std_new = np.std(self.raw_recorder[self.samples_counter - self.freq :
                                                                    self.samples_counter], 0)
                        if self.raw_std is None:
                            self.raw_std = raw_std_new
                        else:
                            if self.catch_channels_trouble and any(raw_std_new > 7 * self.raw_std):
                                w = ChannelTroubleWarning(parent=self.main)
                                w.pause_clicked.connect(self.handle_channels_trouble_pause)
                                w.closed.connect(
                                    lambda: self.enable_trouble_catching(w)
                                )
                                w.show()
                                self.catch_channels_trouble = False
                            self.raw_std = 0.5 * raw_std_new + 0.5 * self.raw_std



            # redraw signals and raw data
            self.main.redraw_signals(self.current_samples, chunk, self.samples_counter)
            # redraw protocols
            is_half_time = self.samples_counter >= self.current_protocol_n_samples // 2
            current_protocol = self.protocols_sequence[self.current_protocol_index]
            if current_protocol.mock_previous > 0:
                samples = [signal.current_sample for signal in current_protocol.mock]
            elif current_protocol.mock_samples_file_path is not None:
                samples = self.mock_signals_buffer[self.samples_counter % self.mock_signals_buffer.shape[0]]
            else:
                samples = self.current_samples

            self.reward.update(samples[self.reward.signal_ind], chunk.shape[0])
            if (self.main.player_panel.start.isChecked() and
                self.samples_counter - chunk.shape[0] < self.experiment_n_samples):
                self.reward_recorder[self.samples_counter-chunk.shape[0]:self.samples_counter] = self.reward.get_score()

            # subject update
            self.subject.update_protocol_state(samples, chunk_size=chunk.shape[0], is_half_time=is_half_time)

            # change protocol if current_protocol_n_samples has been reached
            if self.samples_counter >= self.current_protocol_n_samples and not self.test_mode:
                self.next_protocol()

    def enable_trouble_catching(self, widget):
        self.catch_channels_trouble = not widget.ignore_flag

    def start_test_protocol(self, protocol):
        print('Experiment: test')
        if not self.main_timer.isActive():
            self.main_timer.start(1000 * 1. / self.freq)
        self.samples_counter = 0
        self.main.signals_buffer *= 0
        self.test_mode = True

        self.subject.change_protocol(protocol)


    def close_test_protocol(self):
        if self.main_timer.isActive():
            self.main_timer.stop()
        self.samples_counter = 0
        self.main.signals_buffer *= 0
        self.test_mode = False

    def handle_channels_trouble_pause(self):
        print('pause clicked')
        if self.main.player_panel.start.isChecked():
            self.main.player_panel.start.click()

    def handle_channels_trouble_continue(self, pause_enabled):
        print('continue clicked')
        if not pause_enabled and not self.main.player_panel.start.isChecked():
            self.main.player_panel.start.click()

    def next_protocol(self):
        """
        Change protocol
        :return: None
        """
        # save raw and signals samples asynchronously
        protocol_number_str = 'protocol' + str(self.current_protocol_index + 1)

        # close previous protocol
        self.protocols_sequence[self.current_protocol_index].close_protocol(
            raw=self.raw_recorder[:self.samples_counter],
            signals=self.signals_recorder[:self.samples_counter],
            protocols=self.protocols)

        save_signals(self.dir_name + 'experiment_data.h5', self.signals, protocol_number_str,
                     raw_data=self.raw_recorder[:self.samples_counter],
                     raw_other_data=self.raw_recorder_other[:self.samples_counter],
                     signals_data=self.signals_recorder[:self.samples_counter],
                     reward_data=self.reward_recorder[:self.samples_counter])

        # reset samples counter
        previous_counter = self.samples_counter
        self.samples_counter = 0
        self.seconds = self.freq

        # reset buffer if previous protocol has true value in update_statistics_in_the_end
        if self.protocols_sequence[self.current_protocol_index].update_statistics_in_the_end:
            self.main.signals_buffer *= 0
        self.main.update_statistics_lines()

        if self.current_protocol_index < len(self.protocols_sequence) - 1:

            # update current protocol index and n_samples
            self.current_protocol_index += 1
            current_protocol = self.protocols_sequence[self.current_protocol_index]
            self.current_protocol_n_samples = self.freq * current_protocol.duration

            # prepare mock from raw if necessary
            if current_protocol.mock_previous:
                if current_protocol.mock_previous == self.current_protocol_index:
                    mock_raw = self.raw_recorder[:previous_counter]
                else:
                    mock_raw = load_h5py(self.dir_name + 'experiment_data.h5',
                                         'protocol{}/raw_data'.format(current_protocol.mock_previous))
                current_protocol.prepare_raw_mock_if_necessary(mock_raw)

            # change protocol widget
            self.subject.change_protocol(current_protocol)
            if current_protocol.mock_samples_file_path is not None:
                self.mock_signals_buffer = load_h5py_protocol_signals(
                    current_protocol.mock_samples_file_path,
                    current_protocol.mock_samples_protocol)
            self.main.status.update()

            self.reward.threshold = current_protocol.reward_threshold
            reward_signal_id = current_protocol.reward_signal_id
            self.reward.signal = self.signals[reward_signal_id]  # TODO: REward for MOCK
            self.reward.set_enabled(isinstance(current_protocol, FeedbackProtocol))

        else:
            # status
            self.main.status.finish()
            # action in the end of protocols sequence
            self.current_protocol_n_samples = np.inf
            self.is_finished = True
            self.subject.close()
            # np.save('results/raw', self.main.raw_recorder)
            # np.save('results/signals', self.main.signals_recorder)

            # save_h5py(self.dir_name + 'raw.h5', self.main.raw_recorder)
            # save_h5py(self.dir_name + 'signals.h5', self.main.signals_recorder)
            params_to_xml_file(self.params, self.dir_name + 'settings.xml')
            self.stream.save_info(self.dir_name + 'lsl_stream_info.xml')

    def restart(self):
        wait_bar = WaitMessage('Experiment is starting. Please wait ...').show_and_return()

        self.test_mode = False
        if self.main_timer is not None:
            self.main_timer.stop()
        if self.stream is not None:
            self.stream.disconnect()
        if self.thread is not None:
            self.thread.terminate()

        # timer
        self.main_timer = QtCore.QTimer(self.app)

        self.is_finished = False

        # current protocol index
        self.current_protocol_index = 0

        # samples counter for protocol sequence
        self.samples_counter = 0

        # run raw
        self.thread = None
        if self.params['sInletType'] == 'lsl_from_file':
            self.restart_lsl_from_file()
        elif self.params['sInletType'] == 'lsl_generator':
            self.thread = Process(target=run_eeg_sim, args=(),
                                  kwargs={'chunk_size': 0, 'name': self.params['sStreamName']})
            self.thread.start()
            from time import sleep
            sleep(2)
        if self.params['sInletType'] == 'ftbuffer':
            hostname, port = self.params['sFTHostnamePort'].split(':')
            port = int(port)
            stream = FieldTripBufferInlet(hostname, port)
        else:
            stream = LSLInlet(name=self.params['sStreamName'])
        self.stream = ChannelsSelector(stream, exclude=self.params['sReference'])
        self.freq = self.stream.get_frequency()
        self.n_channels = self.stream.get_n_channels()
        self.n_channels_other = self.stream.get_n_channels_other()
        channels_labels = self.stream.get_channels_labels()

        self.seconds = 2 * self.freq
        self.raw_std = None

        # signals
        self.signals = [DerivedSignal(ind=ind,
                                      bandpass_high=signal['fBandpassHighHz'],
                                      bandpass_low=signal['fBandpassLowHz'],
                                      name=signal['sSignalName'],
                                      n_channels=self.n_channels,
                                      spatial_filter=(read_spatial_filter(signal['SpatialFilterMatrix'],
                                                                          channels_labels)
                                                      if signal['SpatialFilterMatrix'] != ''
                                                      else None),
                                      disable_spectrum_evaluation=signal['bDisableSpectrumEvaluation'],
                                      n_samples=signal['fFFTWindowSize'],
                                      smoothing_factor=signal['fSmoothingFactor'])
                        for ind, signal in enumerate(self.params['vSignals']['DerivedSignal'])]

        # composite signals
        self.composite_signals = [CompositeSignal([s for s in self.signals],
                                                  signal['sExpression'],
                                                  signal['sSignalName'],
                                                  ind=ind + len(self.signals))
                                  for ind, signal in enumerate(self.params['vSignals']['CompositeSignal'])]
        self.signals += self.composite_signals
        self.current_samples = np.zeros_like(self.signals)

        # protocols
        self.protocols = []
        signal_names = [signal.name for signal in self.signals]

        for protocol in self.params['vProtocols']:
            source_signal_id = None if protocol['fbSource'] == 'All' else signal_names.index(protocol['fbSource'])
            reward_signal_id = signal_names.index(protocol['sRewardSignal']) if protocol['sRewardSignal'] != '' else 0
            print(protocol['sRewardSignal'], reward_signal_id)
            mock_path = (protocol['sMockSignalFilePath'] if protocol['sMockSignalFilePath'] != '' else None,
                         protocol['sMockSignalFileDataset'])
            if protocol['sFb_type'] == 'Baseline':
                self.protocols.append(
                    BaselineProtocol(
                        self.signals,
                        duration=protocol['fDuration'],
                        name=protocol['sProtocolName'],
                        source_signal_id=source_signal_id,
                        text=protocol['cString'] if protocol['cString'] != '' else 'Relax',
                        update_statistics_in_the_end=bool(protocol['bUpdateStatistics']),
                        drop_outliers=int(protocol['iDropOutliers']),
                        ssd_in_the_end=bool(protocol['bSSDInTheEnd']),
                        freq=self.freq,
                        timer=self.main_timer,
                        ch_names=channels_labels,
                        show_reward=bool(protocol['bShowReward']),
                        reward_threshold=protocol['bRewardThreshold'],
                        reward_signal_id=reward_signal_id,
                        half_time_text=protocol['cString2'] if bool(protocol['bUseExtraMessage']) else None,
                        experiment=self
                    ))
            elif protocol['sFb_type'] == 'CircleFeedback':
                self.protocols.append(
                    FeedbackProtocol(
                        self.signals,
                        duration=protocol['fDuration'],
                        name=protocol['sProtocolName'],
                        source_signal_id=source_signal_id,
                        mock_samples_path=mock_path,
                        mock_previous=int(protocol['iMockPrevious']),
                        update_statistics_in_the_end=bool(protocol['bUpdateStatistics']),
                        drop_outliers=int(protocol['iDropOutliers']),
                        ssd_in_the_end=bool(protocol['bSSDInTheEnd']),
                        freq=self.freq,
                        timer=self.main_timer,
                        ch_names=channels_labels,
                        show_reward=bool(protocol['bShowReward']),
                        reward_threshold=protocol['bRewardThreshold'],
                        reward_signal_id=reward_signal_id,
                        experiment=self))
            elif protocol['sFb_type'] == 'ThresholdBlink':
                self.protocols.append(
                    ThresholdBlinkFeedbackProtocol(
                        self.signals,
                        duration=protocol['fDuration'],
                        name=protocol['sProtocolName'],
                        threshold=protocol['fBlinkThreshold'],
                        time_ms=protocol['fBlinkDurationMs'],
                        freq=self.freq,
                        timer=self.main_timer,
                        source_signal_id=source_signal_id,
                        ch_names=channels_labels,
                        update_statistics_in_the_end=bool(protocol['bUpdateStatistics']),
                        drop_outliers=int(protocol['iDropOutliers']),
                        ssd_in_the_end=bool(protocol['bSSDInTheEnd']),
                        show_reward=bool(protocol['bShowReward']),
                        reward_threshold=protocol['bRewardThreshold'],
                        reward_signal_id=reward_signal_id,
                        experiment=self))
            else:
                raise TypeError('Undefined protocol type \"{}\"'.format(protocol['sFb_type']))

        # protocols sequence
        names = [protocol.name for protocol in self.protocols]
        self.protocols_sequence = []
        for name in self.params['vPSequence']:
            self.protocols_sequence.append(self.protocols[names.index(name)])

        # reward
        from pynfb.reward import Reward
        self.reward = Reward(self.protocols[0].reward_signal_id,
                             threshold=self.protocols[0].reward_threshold,
                             rate_of_increase=self.params['fRewardPeriodS'],
                             fs=self.freq)

        self.reward.set_enabled(isinstance(self.protocols_sequence[0], FeedbackProtocol))

        # timer
        # self.main_timer = QtCore.QTimer(self.app)
        self.main_timer.timeout.connect(self.update)
        self.main_timer.start(1000 * 1. / self.freq)

        # current protocol number of samples ('frequency' * 'protocol duration')
        self.current_protocol_n_samples = self.freq * self.protocols_sequence[self.current_protocol_index].duration

        # experiment number of samples
        max_protocol_n_samples = max([self.freq * p.duration for p in self.protocols_sequence])

        # data recorders
        self.experiment_n_samples = max_protocol_n_samples
        self.samples_counter = 0
        self.raw_recorder = np.zeros((max_protocol_n_samples * 110 // 100, self.n_channels)) * np.nan
        self.raw_recorder_other = np.zeros((max_protocol_n_samples * 110 // 100, self.n_channels_other)) * np.nan
        self.signals_recorder = np.zeros((max_protocol_n_samples * 110 // 100, len(self.signals))) * np.nan
        self.reward_recorder = np.zeros((max_protocol_n_samples * 110 // 100)) * np.nan

        # save init signals
        save_signals(self.dir_name + 'experiment_data.h5', self.signals,
                     group_name='protocol0')

        # windows
        self.main = MainWindow(signals=self.signals,
                               protocols=self.protocols_sequence,
                               parent=None,
                               experiment=self,
                               current_protocol=self.protocols_sequence[self.current_protocol_index],
                               n_signals=len(self.signals),
                               max_protocol_n_samples=max_protocol_n_samples,
                               freq=self.freq,
                               n_channels=self.n_channels,
                               plot_raw_flag=self.params['bPlotRaw'],
                               plot_signals_flag=self.params['bPlotSignals'],
                               channels_labels=channels_labels)
        self.subject = self.main.subject_window

        if self.params['sInletType'] == 'lsl_from_file':
            self.main.player_panel.start_clicked.connect(self.restart_lsl_from_file)

        wait_bar.close()

    def restart_lsl_from_file(self):
        if self.thread is not None:
            self.thread.terminate()
        source_buffer = load_h5py_all_samples(self.params['sRawDataFilePath']).T
        self.thread = Process(target=run_eeg_sim, args=(),
                              kwargs={'chunk_size': 0, 'source_buffer': source_buffer,
                                      'name': self.params['sStreamName']})
        self.thread.start()
        from time import sleep
        sleep(2)

    def destroy(self):
        if self.thread is not None:
            self.thread.terminate()
        self.main_timer.stop()
        del self.stream
        self.stream = None
        # del self
