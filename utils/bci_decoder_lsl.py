import os
import sys
import logging
import numpy as np
import scipy.signal as sg
from datetime import datetime
from psychopy import visual
from psychopy.visual.circle import Circle
from psychopy.sound import Sound
from pylsl import StreamInlet, resolve_byprop
from pylsl.pylsl import lib, StreamInfo, FOREVER, c_int, c_double, byref, handle_error, StreamOutlet
import xml.etree.ElementTree as ET
from pynfb.inlets.lsl_inlet import LSLInlet
from pynfb.signal_processing.filters import CFIRBandEnvelopeDetector, IdentityFilter, ExponentialSmoother
from pynfb.signals.bci import BCIModel


class Visualizer:
    def __init__(self):
        # create a window
        self.win = visual.Window([800, 600], monitor="testMonitor", units="deg")
        self.text_stim = visual.TextStim(self.win, text='+++++++++++++++++')
        self.fb_stim = Circle(self.win)
        self.sound = Sound('A', 0.3)
        pass

    def set_message(self, message):
        self.text_stim.setText(message)
        self.text_stim.draw()
        self.win.update()
        # print('Monitor Message="{}"'.format(message), end='\r')

    def beep(self):
        self.sound.play()


    def update_fb_stim(self, val):
        self.fb_stim.radius = min(max(val, 0), 1)*1 + 1
        self.fb_stim.draw()
        self.win.update()
        # print('Monitor FB="{}"'.format(val), end="\r")

    def close(self):
        self.win.close()


# record data
def record_data(exp_settings, inlet, visualizer):
    fs = int(inlet.get_frequency())
    channels = inlet.get_channels_labels()
    n_channels = len(channels)
    logging.info('Stream channels = {}'.format(np.array(channels)))

    # prepare recording utils
    block_durations = [exp_settings['blocks'][block_name]['duration']
                       for block_name in exp_settings['sequence']]
    n_seconds = sum(block_durations)
    n_blocks = len(exp_settings['sequence'])
    buffer = np.empty((n_seconds * fs + 100 * fs, n_channels + 1))

    STIM_CHANNEL = n_channels

    n_samples_received = 0
    for j_block, block_name in enumerate(exp_settings['sequence']):
        current_block = exp_settings['blocks'][block_name]
        n_samples = fs * current_block['duration']
        block_id = current_block['id']
        logging.info('Block #{}/{} id{} "{}" ({}s)'.format(
            j_block, n_blocks, block_id, block_name, current_block['duration']))
        n_samples_received_in_block = 0
        while n_samples_received_in_block < n_samples:
            chunk, t_stamp = inlet.get_next_chunk()

            if chunk is not None:
                n_samples_in_chunk = len(chunk)
                if 'message' in current_block:
                    visualiser.set_message(current_block['message'])

                buffer[n_samples_received:n_samples_received + n_samples_in_chunk,
                :n_channels] = chunk
                buffer[n_samples_received:n_samples_received + n_samples_in_chunk,
                STIM_CHANNEL] = block_id
                n_samples_received_in_block += n_samples_in_chunk
                n_samples_received += n_samples_in_chunk

    # save recorded data
    recorded_data = buffer[:n_samples_received]
    visualiser.close()
    return recorded_data, channels, fs


class BCIStreamOutlet(StreamOutlet):
    def __init__(self, name, fs):
        # create stream info
        info = StreamInfo(name=name, type='BCI', channel_count=1, nominal_srate=fs,
                          channel_format='float32', source_id='myuid34234')

        # set channels labels (in accordance with XDF format, see also code.google.com/p/xdf)
        chns = info.desc().append_child("channels")
        ch = chns.append_child("channel")
        ch.append_child_value("label", 'STATE')

        # init stream
        super(BCIStreamOutlet, self).__init__(info)


if __name__ == '__main__':
    # create results folder
    timestamp_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
    results_path = 'results/{}_{}/'.format('bci_exp', timestamp_str)
    os.makedirs(results_path)

    # setup logger
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(results_path, 'log.txt'))]
    logging.basicConfig(format='%(asctime)s> %(message)s', level=logging.INFO, handlers=handlers)

    # main parameters
    PROBES_DURATION = 3
    STATES_CODES_DICT = {'Rest': 2, 'Left': 3, 'Right':4 , 'Legs': 0}
    EXP_SETTINGS_PROBES = {
        'exp_name': 'example',
        'lsl_stream_name': 'lsl_sim',
        'blocks': {
            'Rest': {'duration': PROBES_DURATION, 'id': STATES_CODES_DICT['Rest'], 'message': '+'},
            'Legs': {'duration': PROBES_DURATION, 'id': STATES_CODES_DICT['Legs'], 'message': 'Ноги'},
            'Right': {'duration': PROBES_DURATION, 'id': STATES_CODES_DICT['Right'], 'message': 'Правая рука'},
            'Left': {'duration': PROBES_DURATION, 'id': STATES_CODES_DICT['Left'], 'message': 'Левая рука'},
            'Prepare': {'duration': 10, 'id': 42, 'message': 'Готовность 10 секунд'}
        },
        'sequence': ['Rest', 'Left', 'Right']*10,
    }

    # connect to LSL stream
    inlet = LSLInlet(EXP_SETTINGS_PROBES['lsl_stream_name'])

    # visualizer
    visualiser = Visualizer()

    # record probes
    recorded_data, channels, fs = record_data(EXP_SETTINGS_PROBES, inlet, visualiser)
    # np.savez(results_path + 'probes.npz', data=recorded_data, channels=channels + ['STIM'], fs=fs)
    # logging.info('Saved to {}'.format(results_path + 'probes.npz'))
    # import pylab as plt
    # plt.plot(recorded_data[:, -2:])
    # plt.show()

    # fit bci model
    states = [STATES_CODES_DICT[state] for state in np.unique(EXP_SETTINGS_PROBES['sequence'])]
    print(states)
    bci_model = BCIModel(fs, [(8, 15), (18, 28)], channels, states, [0, -1])
    bci_model.fit(recorded_data[:, :-1], recorded_data[:, -1])

    stream = BCIStreamOutlet('BCI_state', fs)
    # run bci model
    while True:
        chunk, t_stamp = inlet.get_next_chunk()
        if chunk is not None:
            states = bci_model.apply(chunk)
            stream.push_chunk(states.tolist())
            print(bci_model.apply(chunk)[-1])
