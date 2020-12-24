from pynfb.helpers.simple_socket import SimpleServer
from pynfb.inlets.lsl_inlet import LSLInlet
from time import sleep
from psychopy import visual, core
import numpy as np
from pynfb.signal_processing.filters import CFIRBandEnvelopeDetector, DownsampleFilter, SpatialFilter, IdentityFilter
from pynfb.outlets.signals_outlet import SignalsOutlet

# init psychopy window
win = visual.Window([600, 600])
message = visual.TextStim(win, text='', alignHoriz='center')
message.autoDraw = True  # Automatically draw every frame

# connect to LSL inlet
message.text = 'Сообщение экспериментатору:\nПодключение к LSL потоку "NVX136_Data"...'
win.flip()
lsl_in = LSLInlet('NVX136_Data')
fs = int(lsl_in.get_frequency())
channels = lsl_in.get_channels_labels()

# create LSL outlet
lsl_out = SignalsOutlet(channels, fs=500, name='NVX136_FB')

# connect to NFBLab
message.text = 'Сообщение экспериментатору:\nЗапустите эксперимент в NFBLab'
win.flip()
server = SimpleServer()

# setup filters
downsampler = DownsampleFilter(10, len(channels))
spatial_filter = np.zeros(len(channels))
cfir = CFIRBandEnvelopeDetector([8, 12], fs, IdentityFilter())
mean = 0
std = 1


while 1:
    chunk, timestamp = lsl_in.get_next_chunk()
    if chunk is not None:
        print(chunk.shape)
        chunk = downsampler.apply(chunk)

        if len(chunk) > 0:
            print(chunk.shape)
            virtual_channel = chunk.dot(spatial_filter)
            envelope = cfir.apply(virtual_channel)
            score = (envelope - mean)/std if std > 0 else 404

            for sample in chunk:
                lsl_out.push_sample(sample)

    meta_str, obj = server.pull_message()
    if meta_str == 'msg':
        print('Dummy.. Set message to "{}"'.format(obj))
        message.text = obj
        win.flip()
    if meta_str == 'fb1':
        print('Dummy.. Run FB. Set message to "{}"'.format(obj))
        message.text = obj
        win.flip()
    if meta_str == 'spt':
        spatial_filter = obj
        print('Dummy.. Set spatial filter to {}'.format(obj))
    if meta_str == 'bnd':
        cfir = CFIRBandEnvelopeDetector(obj, fs, IdentityFilter())
        print('Dummy.. Set band to {}'.format(obj))
    if meta_str == 'std':
        mean, std = obj
        print('Dummy.. Set stats to {}'.format(obj))



