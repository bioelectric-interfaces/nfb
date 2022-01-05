import os
import re
from datetime import datetime
import random as r
import numpy as np
from PyQt5 import QtCore
from itertools import zip_longest, chain

from pynfb.inlets.montage import Montage
from pynfb.postprocessing.plot_all_fb_bars import plot_fb_dynamic
from pynfb.widgets.channel_trouble import ChannelTroubleWarning
from pynfb.widgets.helpers import WaitMessage
from pynfb.outlets.signals_outlet import SignalsOutlet
from .generators import run_eeg_sim, stream_file_in_a_thread, stream_generator_in_a_thread
from .inlets.ftbuffer_inlet import FieldTripBufferInlet
from .inlets.lsl_inlet import LSLInlet
from .inlets.channels_selector import ChannelsSelector
from .serializers.hdf5 import save_h5py, load_h5py, save_signals, load_h5py_protocol_signals, save_xml_str_to_hdf5_dataset, \
    save_channels_and_fs
from .serializers.xml_ import params_to_xml_file, params_to_xml, get_lsl_info_from_xml
from .serializers import read_spatial_filter
from .protocols import BaselineProtocol, FeedbackProtocol, ThresholdBlinkFeedbackProtocol, VideoProtocol, \
    ParticipantInputProtocol, ParticipantChoiceProtocol, ExperimentStartProtocol, FixationCrossProtocol, ImageProtocol, \
    GaborFeedbackProtocolWidgetPainter, ParticipantChoiceWidgetPainter
from .signals import DerivedSignal, CompositeSignal, BCISignal
from .windows import MainWindow
from ._titles import WAIT_BAR_MESSAGES
import mne


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
        self.mock_signals_buffer = None
        self.activate_trouble_catching = False
        self.main = None
        self.gabor_theta = r.choice(range(0, 360, 30)) # Init the gabor theta with a random angle
        self.rn_offset = r.choice([-5, 5, 0, 0]) # Init Random offset between +/- 5 degrees and 0 for Gabor orientation
        self.mean_reward_signal = 0
        self.fb_score = None
        self.cum_score = None
        self.restart()

        pass

    def update(self):
        """
        Experiment main update action
        :return: None
        """
        # get next chunk
        # self.stream is a ChannelsSelector instance!
        chunk, other_chunk, timestamp = self.stream.get_next_chunk() if self.stream is not None else (None, None)
        if chunk is not None and self.main is not None:

            # update and collect current samples
            for i, signal in enumerate(self.signals):
                signal.update(chunk)
                # self.current_samples[i] = signal.current_sample

            # push current samples
            sample = np.vstack([np.array(signal.current_chunk) for signal in self.signals]).T.tolist()
            self.signals_outlet.push_chunk(sample)

            # record data
            if self.main.player_panel.start.isChecked():
                if self.params['bShowSubjectWindow']:
                    self.subject.figure.update_reward(self.reward.get_score())
                if self.samples_counter < self.experiment_n_samples:
                    chunk_slice = slice(self.samples_counter, self.samples_counter + chunk.shape[0])
                    self.raw_recorder[chunk_slice] = chunk[:, :self.n_channels]
                    self.raw_recorder_other[chunk_slice] = other_chunk
                    self.timestamp_recorder[chunk_slice] = timestamp
                    # for s, sample in enumerate(self.current_samples):
                    self.signals_recorder[chunk_slice] = sample
                    self.samples_counter += chunk.shape[0]

                    # catch channels trouble

                    if self.activate_trouble_catching:
                        if self.samples_counter > self.seconds:
                            self.seconds += 2 * self.freq
                            raw_std_new = np.std(self.raw_recorder[int(self.samples_counter - self.freq):
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
            self.main.redraw_signals(sample, chunk, self.samples_counter, self.current_protocol_n_samples)
            if self.params['bPlotSourceSpace']:
                self.source_space_window.update_protocol_state(chunk)

            # redraw protocols
            is_half_time = self.samples_counter >= self.current_protocol_n_samples // 2
            current_protocol = self.protocols_sequence[self.current_protocol_index]
            if current_protocol.mock_previous > 0:
                samples = [signal.current_chunk[-1] for signal in current_protocol.mock]
            elif current_protocol.mock_samples_file_path is not None:
                samples = self.mock_signals_buffer[self.samples_counter % self.mock_signals_buffer.shape[0]]
            else:
                samples = sample[-1]

            # self.reward.update(samples[self.reward.signal_ind], chunk.shape[0])
            if (self.main.player_panel.start.isChecked() and
                    self.samples_counter - chunk.shape[0] < self.experiment_n_samples):
                self.reward_recorder[
                self.samples_counter - chunk.shape[0]:self.samples_counter] = self.reward.get_score()

            if self.main.player_panel.start.isChecked():
                # subject update
                if self.params['bShowSubjectWindow']:
                    mark = self.subject.update_protocol_state(samples, self.reward, chunk_size=chunk.shape[0],
                                                              is_half_time=is_half_time)
                    # if no offset, correct answer is YES, otherwise, correct answer is NO
                    # TODO: make this more generic - i.e. doesn't just depend on rn_offset (gabor theta angle)
                    answer = 1
                    if self.rn_offset:
                        answer = 2
                    # 'yes' response = 1, 'no' response = 2, lack of response = 0
                    choice = current_protocol.response_key
                else:
                    mark = None
                    choice = None
                    answer = None
                self.mark_recorder[self.samples_counter - chunk.shape[0]:self.samples_counter] = 0
                self.mark_recorder[self.samples_counter - 1] = int(mark or 0)

                self.choice_recorder[self.samples_counter - chunk.shape[0]:self.samples_counter] = 0
                self.choice_recorder[self.samples_counter - 1] = int(choice or 0)
                self.answer_recorder[self.samples_counter - chunk.shape[0]:self.samples_counter] = 0
                self.answer_recorder[self.samples_counter - 1] = int(answer or 0)

            # If probe, display probe at random time after beginning of delay
            if current_protocol.show_probe:
                current_protocol.widget_painter.probe = True
                current_protocol.widget_painter.probe_loc = r.choice(["RIGHT", "LEFT"]) # TODO: shift this to 'next protocol' section & init - just like the rn_offset
                pass

            # change protocol if current_protocol_n_samples has been reached
            if self.samples_counter >= self.current_protocol_n_samples and not self.test_mode:
                # If baseline protocol, calculate average of reward signal
                if isinstance(current_protocol, BaselineProtocol):
                    reward_signal_id = current_protocol.reward_signal_id
                    reward_sig = self.signals_recorder[~np.isnan(self.signals_recorder).any(axis=1)]
                    reward_sig = reward_sig[:,reward_signal_id]
                    self.mean_reward_signal = reward_sig.mean()
                    print(f"len signal: {len(reward_sig)}, mean: {reward_sig.mean()}, signal: {reward_sig}")

                # Record the reward from feedback only for the current protocol
                if isinstance(current_protocol, FeedbackProtocol):
                    if self.fb_score:
                        self.fb_score = self.reward.get_score() - self.cum_score
                        self.cum_score = self.reward.get_score()
                    else:
                        self.cum_score = self.reward.get_score()
                        self.fb_score = self.reward.get_score()
                    print(f"fBSCORE: {self.fb_score}, CUMSCORE: {self.reward.get_score()}")

                # only change if not a pausing protocol
                if current_protocol.hold:
                    # don't switch protocols if holding
                    pass
                else:
                    # Reset the hold flag on hold protocols
                    if current_protocol.input_protocol:
                        current_protocol.hold = True
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

        if self.params['bShowSubjectWindow']:
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

        # descale signals:
        signals_recordings = np.array([signal.descale_recording(data)
                                       for signal, data in
                                       zip(self.signals, self.signals_recorder[:self.samples_counter].T)]).T

        # close previous protocol
        self.protocols_sequence[self.current_protocol_index].close_protocol(
            raw=self.raw_recorder[:self.samples_counter],
            signals=signals_recordings,
            protocols=self.protocols,
            protocols_seq=[protocol.name for protocol in self.protocols_sequence[:self.current_protocol_index + 1]],
            raw_file=self.dir_name + 'experiment_data.h5',
            marks=self.mark_recorder[:self.samples_counter])

        save_signals(self.dir_name + 'experiment_data.h5', self.signals, protocol_number_str,
                     raw_data=self.raw_recorder[:self.samples_counter],
                     timestamp_data=self.timestamp_recorder[:self.samples_counter],
                     raw_other_data=self.raw_recorder_other[:self.samples_counter],
                     signals_data=signals_recordings,
                     reward_data=self.reward_recorder[:self.samples_counter],
                     protocol_name=self.protocols_sequence[self.current_protocol_index].name,
                     mock_previous=self.protocols_sequence[self.current_protocol_index].mock_previous,
                     mark_data=self.mark_recorder[:self.samples_counter],
                     choice_data=self.choice_recorder[:self.samples_counter],
                     answer_data=self.answer_recorder[:self.samples_counter])

        # reset samples counter
        previous_counter = self.samples_counter
        self.samples_counter = 0
        if self.protocols_sequence[self.current_protocol_index].update_statistics_in_the_end:
            self.main.time_counter1 = 0
            self.main.signals_viewer.reset_buffer()
        self.seconds = self.freq

        # list of real fb protocols (number in protocol sequence)
        if isinstance(self.protocols_sequence[self.current_protocol_index], FeedbackProtocol):
            if self.protocols_sequence[self.current_protocol_index].mock_previous == 0:
                self.real_fb_number_list += [self.current_protocol_index + 1]
        elif self.protocols_sequence[self.current_protocol_index].as_mock:
            self.real_fb_number_list += [self.current_protocol_index + 1]

        if self.current_protocol_index < len(self.protocols_sequence) - 1:

            # update current protocol index and n_samples
            self.current_protocol_index += 1
            current_protocol = self.protocols_sequence[self.current_protocol_index]
            self.current_protocol_n_samples = self.freq * (
                    self.protocols_sequence[self.current_protocol_index].duration +
                    np.random.uniform(0, self.protocols_sequence[self.current_protocol_index].random_over_time))

            # Reset participant key
            current_protocol.response_key = None
            # Update gabor patch angle for next gabor
            # TODO: make this more generic (only dependant on the protocol)
            bc_threshold = None
            if isinstance(current_protocol.widget_painter, GaborFeedbackProtocolWidgetPainter):
                self.gabor_theta = r.choice(range(0, 360, 30))
                print(f"GABOR THETA: {self.gabor_theta}")
                current_protocol.widget_painter.gabor_theta = self.gabor_theta

                # update the threshold for the Gabor feedback protocol with variable percentage
                # TODO: also make this more generic (for all feedback protocols - not just Gabor)
                reward_bound = 0.25 # percent to add to the bias # TODO: make this a GUI flag
                # TODO: how to handle negative bias (currently it makes the test easier if they have a negative bias)
                bc_threshold = self.mean_reward_signal + (reward_bound * self.mean_reward_signal)
                print(f"R THRESHOLD: {bc_threshold}")
                current_protocol.widget_painter.r_threshold = bc_threshold

            # Update the choice gabor angle and score
            if isinstance(current_protocol.widget_painter, ParticipantChoiceWidgetPainter):
                self.rn_offset = r.choice([-5, 5, 0, 0])
                print(f"CHOICE THETA: {self.gabor_theta + self.rn_offset}")
                current_protocol.widget_painter.gabor_theta = self.gabor_theta + self.rn_offset
                current_protocol.widget_painter.previous_score = self.fb_score
                # current_protocol.widget_painter.redraw_state(0,0)

            # prepare mock from raw if necessary
            if current_protocol.mock_previous:
                random_previos_fb = None
                if len(self.real_fb_number_list) > 0:
                    random_previos_fb = self.real_fb_number_list[np.random.randint(0, len(self.real_fb_number_list))]
                if current_protocol.shuffle_mock_previous:
                    current_protocol.mock_previous = random_previos_fb
                print('MOCK from protocol # current_protocol.mock_previous')
                if current_protocol.mock_previous == self.current_protocol_index:
                    mock_raw = self.raw_recorder[:previous_counter]
                    mock_signals = self.signals_recorder[:previous_counter]
                else:
                    mock_raw = load_h5py(self.dir_name + 'experiment_data.h5',
                                         'protocol{}/raw_data'.format(current_protocol.mock_previous))
                    mock_signals = load_h5py(self.dir_name + 'experiment_data.h5',
                                             'protocol{}/signals_data'.format(current_protocol.mock_previous))
                # print(self.real_fb_number_list)

                current_protocol.prepare_raw_mock_if_necessary(mock_raw, random_previos_fb, mock_signals)

            # change protocol widget
            if self.params['bShowSubjectWindow']:
                self.subject.change_protocol(current_protocol)
            if current_protocol.mock_samples_file_path is not None:
                print(f"mockpath: {current_protocol.mock_samples_file_path}, mockprotocol: {current_protocol.mock_samples_protocol}, actual_mock_protocol: protocol{self.current_protocol_index}")
                self.mock_signals_buffer = load_h5py_protocol_signals(
                    current_protocol.mock_samples_file_path,
                    f"protocol{self.current_protocol_index}") # TODO: [ ]fix this - it only works if there are the same number of protocols in sham and real (study must be identical)
                    # current_protocol.mock_samples_protocol)
            self.main.status.update()

            if bc_threshold:
                # if using a baseline-corrected threshold
                self.reward.threshold = bc_threshold
            else:
                self.reward.threshold = current_protocol.reward_threshold
            reward_signal_id = current_protocol.reward_signal_id
            print(self.signals)
            print(reward_signal_id)
            if current_protocol.mock_samples_file_path is not None:
                self.reward.signal = self.mock_signals_buffer[reward_signal_id]
            else:
                self.reward.signal = self.signals[reward_signal_id]  # TODO: REward for MOCK
            self.reward.set_enabled(isinstance(current_protocol, FeedbackProtocol))

        else:
            # status
            self.main.status.finish()
            # action in the end of protocols sequence
            self.current_protocol_n_samples = np.inf
            self.is_finished = True
            if self.params['bShowSubjectWindow']:
                self.subject.close()
            if self.params['bPlotSourceSpace']:
                self.source_space_window.close()
            # plot_fb_dynamic(self.dir_name + 'experiment_data.h5', self.dir_name)
            # np.save('results/raw', self.main.raw_recorder)
            # np.save('results/signals', self.main.signals_recorder)

            # save_h5py(self.dir_name + 'raw.h5', self.main.raw_recorder)
            # save_h5py(self.dir_name + 'signals.h5', self.main.signals_recorder)

    def restart(self):

        timestamp_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
        self.dir_name = 'results/{}_{}/'.format(self.params['sExperimentName'], timestamp_str)
        os.makedirs(self.dir_name)

        wait_bar = WaitMessage(WAIT_BAR_MESSAGES['EXPERIMENT_START']).show_and_return()

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

        # run file lsl stream in a thread
        self.thread = None
        if self.params['sInletType'] == 'lsl_from_file':
            self.restart_lsl_from_file()

        # run simulated eeg lsl stream in a thread
        elif self.params['sInletType'] == 'lsl_generator':
            self.thread = stream_generator_in_a_thread(self.params['sStreamName'])

        # use FTB inlet
        aux_streams = None
        if self.params['sInletType'] == 'ftbuffer':
            hostname, port = self.params['sFTHostnamePort'].split(':')
            port = int(port)
            stream = FieldTripBufferInlet(hostname, port)

        # use LSL inlet
        else:
            stream_names = re.split(r"[,;]+", self.params['sStreamName'])
            print(f'STREAM NAME: {stream_names}')
            streams = [LSLInlet(name=name) for name in stream_names]
            stream = streams[0]
            aux_streams = streams[1:] if len(streams) > 1 else None

        # setup events stream by name
        events_stream_name = self.params['sEventsStreamName']
        events_stream = LSLInlet(events_stream_name) if events_stream_name else None

        # setup main stream
        self.stream = ChannelsSelector(stream, exclude=self.params['sReference'],
                                       subtractive_channel=self.params['sReferenceSub'],
                                       dc=self.params['bDC'], events_inlet=events_stream, aux_inlets=aux_streams,
                                       prefilter_band=self.params['sPrefilterBand'])
        self.stream.save_info(self.dir_name + 'stream_info.xml')
        save_channels_and_fs(self.dir_name + 'experiment_data.h5', self.stream.get_channels_labels(),
                             self.stream.get_frequency())

        save_xml_str_to_hdf5_dataset(self.dir_name + 'experiment_data.h5', self.stream.info_as_xml(), 'stream_info.xml')
        self.freq = self.stream.get_frequency()
        self.n_channels = self.stream.get_n_channels()
        self.n_channels_other = self.stream.get_n_channels_other()
        channels_labels = self.stream.get_channels_labels()
        montage = Montage(channels_labels)
        print(montage)
        self.seconds = 2 * self.freq
        self.raw_std = None

        # signals
        self.signals = [DerivedSignal.from_params(ind, self.freq, self.n_channels, channels_labels, signal)
                        for ind, signal in enumerate(self.params['vSignals']['DerivedSignal']) if
                        not signal['bBCIMode']]

        # composite signals
        self.composite_signals = [CompositeSignal([s for s in self.signals],
                                                  signal['sExpression'],
                                                  signal['sSignalName'],
                                                  ind + len(self.signals), self.freq)
                                  for ind, signal in enumerate(self.params['vSignals']['CompositeSignal'])]

        # bci signals
        self.bci_signals = [BCISignal(self.freq, channels_labels, signal['sSignalName'], ind)
                            for ind, signal in enumerate(self.params['vSignals']['DerivedSignal']) if
                            signal['bBCIMode']]

        self.signals += self.composite_signals
        self.signals += self.bci_signals
        # self.current_samples = np.zeros_like(self.signals)

        # signals outlet
        self.signals_outlet = SignalsOutlet([signal.name for signal in self.signals], fs=self.freq)

        # protocols
        self.protocols = []
        signal_names = [signal.name for signal in self.signals]

        for protocol in self.params['vProtocols']:
            # some general protocol arguments
            source_signal_id = None if protocol['fbSource'] == 'All' else signal_names.index(protocol['fbSource'])
            reward_signal_id = signal_names.index(protocol['sRewardSignal']) if protocol['sRewardSignal'] != '' else 0
            mock_path = (protocol['sMockSignalFilePath'] if protocol['sMockSignalFilePath'] != '' else None,
                         protocol['sMockSignalFileDataset'])
            m_signal = protocol['sMSignal']
            m_signal_index = None if m_signal not in signal_names else signal_names.index(m_signal)

            # general protocol arguments dictionary
            kwargs = dict(
                source_signal_id=source_signal_id,
                name=protocol['sProtocolName'],
                duration=protocol['fDuration'],
                random_over_time=protocol['fRandomOverTime'],
                update_statistics_in_the_end=bool(protocol['bUpdateStatistics']),
                stats_type=protocol['sStatisticsType'],
                mock_samples_path=mock_path,
                show_reward=bool(protocol['bShowReward']),
                reward_signal_id=reward_signal_id,
                reward_threshold=protocol['bRewardThreshold'],
                ssd_in_the_end=protocol['bSSDInTheEnd'],
                timer=self.main_timer,
                freq=self.freq,
                mock_previous=int(protocol['iMockPrevious']),
                drop_outliers=int(protocol['iDropOutliers']),
                experiment=self,
                pause_after=bool(protocol['bPauseAfter']),
                beep_after=bool(protocol['bBeepAfter']),
                reverse_mock_previous=bool(protocol['bReverseMockPrevious']),
                m_signal_index=m_signal_index,
                shuffle_mock_previous=bool(protocol['bRandomMockPrevious']),
                as_mock=bool(protocol['bMockSource']),
                auto_bci_fit=bool(protocol['bAutoBCIFit']),
                montage=montage,
                show_probe=protocol['bProbe']
            )

            # type specific arguments
            if protocol['sFb_type'] == 'Baseline':
                self.protocols.append(
                    BaselineProtocol(
                        self.signals,
                        text=protocol['cString'] if protocol['cString'] != '' else 'Relax',
                        half_time_text=protocol['cString2'] if bool(protocol['bUseExtraMessage']) else None,
                        voiceover=protocol['bVoiceover'], **kwargs
                    ))
            elif protocol['sFb_type'] in ['Feedback', 'CircleFeedback']:
                self.protocols.append(
                    FeedbackProtocol(
                        self.signals,
                        gabor_theta=self.gabor_theta,
                        circle_border=protocol['iRandomBound'],
                        m_threshold=protocol['fMSignalThreshold'],
                        **kwargs))
            elif protocol['sFb_type'] == 'ThresholdBlink':
                self.protocols.append(
                    ThresholdBlinkFeedbackProtocol(
                        self.signals,
                        threshold=protocol['fBlinkThreshold'],
                        time_ms=protocol['fBlinkDurationMs'],
                        **kwargs))
            elif protocol['sFb_type'] == 'FixationCross':
                colour_dict = {'Black': (0, 0, 0), 'White': (255, 255, 255), 'Green': (0, 255, 0), 'Red': (255, 0, 0),
                               'Blue': (0, 0, 255)}
                self.protocols.append(
                    FixationCrossProtocol(
                        self.signals,
                        colour=colour_dict[protocol['tFixationCrossColour']],
                        **kwargs))
            elif protocol['sFb_type'] == 'Video':
                self.protocols.append(
                    VideoProtocol(
                        self.signals,
                        video_path=protocol['sVideoPath'],
                        **kwargs))
            elif protocol['sFb_type'] == 'Image':
                self.protocols.append(
                    ImageProtocol(
                        self.signals,
                        image_path=protocol['sVideoPath'],
                        **kwargs))
            elif protocol['sFb_type'] == 'ParticipantInput':
                self.protocols.append(
                    ParticipantInputProtocol(
                        self.signals,
                        text=protocol['cString'] if protocol['cString'] != '' else 'Relax',
                        **kwargs))
            elif protocol['sFb_type'] == 'ParticipantChoice':
                self.protocols.append(
                    ParticipantChoiceProtocol(
                        self.signals,
                        gabor_theta= self.gabor_theta + self.rn_offset,
                        text=protocol['cString'] if protocol['cString'] != '' else 'Relax',
                        **kwargs))
            elif protocol['sFb_type'] == 'ExperimentStart':
                self.protocols.append(
                    ExperimentStartProtocol(
                        self.signals,
                        text=protocol['cString'] if protocol['cString'] != '' else 'Relax',
                        **kwargs))
            else:
                raise TypeError('Undefined protocol type \"{}\"'.format(protocol['sFb_type']))


        # protocols sequence
        names = [protocol.name for protocol in self.protocols]
        group_names = [p['sName'] for p in self.params['vPGroups']['PGroup']]
        print(group_names)
        self.protocols_sequence = []
        for name in self.params['vPSequence']:
            if name in names:
                self.protocols_sequence.append(self.protocols[names.index(name)])
            if name in group_names:
                group = self.params['vPGroups']['PGroup'][group_names.index(name)]
                subgroup = []
                if len(group['sList'].split(' ')) == 1:
                    subgroup.append([group['sList']] * int(group['sNumberList']))
                else:
                    for s_name, s_n in zip(group['sList'].split(' '), list(map(int, group['sNumberList'].split(' ')))):
                        subgroup.append([s_name] * s_n)
                if group['bShuffle']:
                    subgroup = np.concatenate(subgroup)
                    subgroup = list(subgroup[np.random.permutation(len(subgroup))])
                else:
                    subgroup = [k for k in chain(*zip_longest(*subgroup)) if k is not None]
                print(subgroup)
                for subname in subgroup:
                    self.protocols_sequence.append(self.protocols[names.index(subname)])
                    if len(group['sSplitBy']):
                        self.protocols_sequence.append(self.protocols[names.index(group['sSplitBy'])])

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
        self.current_protocol_n_samples = self.freq * (self.protocols_sequence[self.current_protocol_index].duration +
                                                       np.random.uniform(0, self.protocols_sequence[
                                                           self.current_protocol_index].random_over_time))

        # experiment number of samples
        max_protocol_n_samples = int(
            max([self.freq * (p.duration + p.random_over_time) for p in self.protocols_sequence]))

        # data recorders
        self.experiment_n_samples = max_protocol_n_samples
        self.samples_counter = 0
        self.raw_recorder = np.zeros((max_protocol_n_samples * 110 // 100, self.n_channels)) * np.nan
        self.timestamp_recorder = np.zeros((max_protocol_n_samples * 110 // 100)) * np.nan
        self.raw_recorder_other = np.zeros((max_protocol_n_samples * 110 // 100, self.n_channels_other)) * np.nan
        self.signals_recorder = np.zeros((max_protocol_n_samples * 110 // 100, len(self.signals))) * np.nan
        self.reward_recorder = np.zeros((max_protocol_n_samples * 110 // 100)) * np.nan
        self.mark_recorder = np.zeros((max_protocol_n_samples * 110 // 100)) * np.nan
        self.choice_recorder = np.zeros((max_protocol_n_samples * 110 // 100)) * np.nan
        self.answer_recorder = np.zeros((max_protocol_n_samples * 110 // 100)) * np.nan

        # save init signals
        save_signals(self.dir_name + 'experiment_data.h5', self.signals,
                     group_name='protocol0')

        # save settings
        params_to_xml_file(self.params, self.dir_name + 'settings.xml')
        save_xml_str_to_hdf5_dataset(self.dir_name + 'experiment_data.h5', params_to_xml(self.params), 'settings.xml')

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
                               plot_source_space_flag=self.params['bPlotSourceSpace'],
                               show_subject_window=self.params['bShowSubjectWindow'],
                               channels_labels=channels_labels,
                               photo_rect=self.params['bShowPhotoRectangle'])
        self.subject = self.main.subject_window
        if self.params['bPlotSourceSpace']:
            self.source_space_window = self.main.source_space_window

        if self.params['sInletType'] == 'lsl_from_file':
            self.main.player_panel.start_clicked.connect(self.restart_lsl_from_file)


        # create real fb list
        self.real_fb_number_list = []

        wait_bar.close()

    def restart_lsl_from_file(self):
        if self.thread is not None:
            self.thread.terminate()

        file_path = self.params['sRawDataFilePath']
        reference = self.params['sReference']
        stream_name = self.params['sStreamName']

        self.thread = stream_file_in_a_thread(file_path, reference, stream_name)

    def destroy(self):
        if self.thread is not None:
            self.thread.terminate()
        self.main_timer.stop()
        del self.stream
        self.stream = None
        # del self
