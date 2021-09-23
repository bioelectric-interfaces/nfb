from pynfb.helpers.simple_socket import SimpleServer
from pynfb.inlets.lsl_inlet import LSLInlet
from time import sleep
from psychopy import visual, core, sound
import numpy as np
from pynfb.signal_processing.filters import CFIRBandEnvelopeDetector, DownsampleFilter, \
    SpatialFilter, IdentityFilter, ExponentialSmoother, DelayFilter
from pynfb.outlets.signals_outlet import SignalsOutlet
from utils.audiofb.volume_controller import VolumeController
from psychopy.sound.backend_ptb import SoundPTB
import sys

if len(sys.argv) > 1:
    ARTIFICIAL_DELAY_MS = int(sys.argv[1])

else:
    ARTIFICIAL_DELAY_MS = 0
print('\n******\nMOCK FB CONDITION\n******\n'.format(ARTIFICIAL_DELAY_MS))
FS_OUT = 500


# init psychopy window
win = visual.Window(fullscr=False)
message = visual.TextStim(win, text='', alignHoriz='center')
message.autoDraw = True  # Automatically draw every frame
# beep = SoundPTB('A', blockSize=32, volume=1, sampleRate=48000, stereo=False)

# play background music
music = SoundPTB(r'C:\Users\CBI\PycharmProjects\nfb\utils\audiofb\audiocard\backsound.wav', blockSize=32, volume=1)
music.play(loops=100)


# voice synthesizer https://cloud.yandex.ru/services/speechkit#demo
# ogg to wav 1 channel https://online-audio-converter.com/ru/
voices = {'close': SoundPTB('voice/close2.wav', blockSize=32, sampleRate=48000, stereo=True),
          'open': SoundPTB('voice/open_eyes.wav', blockSize=32, sampleRate=48000, stereo=True),
          'filters': SoundPTB('voice/filters.wav', blockSize=32, sampleRate=48000, stereo=True),
          'baseline': SoundPTB('voice/baseline.wav', blockSize=32, sampleRate=48000, stereo=True),
          'pause': SoundPTB('voice/pause.wav', blockSize=32, sampleRate=48000, stereo=True),
          'start': SoundPTB('voice/start.wav', blockSize=32, sampleRate=48000, stereo=True)}
# voices['pause'].play()
# connect to volume controller
message.text = 'Сообщение экспериментатору:\nПодключение к Arduino контроллеру громкости...'
win.flip()
# volume_controller = VolumeController()
sleep(2)
music.volume = 1
# volume_controller.set_volume(100)
sleep(1)
music.volume = 0
voices['pause'].play()
# volume_controller.set_volume(0)


# connect to LSL inlet
message.text = 'Сообщение экспериментатору:\nПодключение к LSL потоку "NVX136_Data"...'
win.flip()
lsl_in = LSLInlet('NVX136_Data')
fs_in = int(lsl_in.get_frequency())
channels = lsl_in.get_channels_labels()

# create LSL outlet
lsl_out = SignalsOutlet(channels + ['FBSIGNAL'], fs=FS_OUT, name='NVX136_FB')

# connect to NFBLab
message.text = 'Сообщение экспериментатору:\nЗапустите эксперимент в NFBLab'
win.flip()
server = SimpleServer()

# setup filters
downsampler = DownsampleFilter(int(fs_in / FS_OUT), len(channels))
spatial_filter = np.zeros(len(channels))
spatial_filter[channels.index('P4')] = 1
cfir = CFIRBandEnvelopeDetector([7, 13], FS_OUT, ExponentialSmoother(0.), n_taps=500)
artificial_delay = DelayFilter(int(ARTIFICIAL_DELAY_MS/1000*FS_OUT))
mean = 0
std = 1

# load mock signal
mock_signal_file = r'C:\Users\CBI\Desktop\audio_nfb\mock_signal.npy'
mock_fb_signal = np.load(mock_signal_file)
current_sample = 0


# init values
score = np.zeros(1)
play_feedback = False
while 1:
    # receive and process chunk
    chunk, timestamp = lsl_in.get_next_chunk()
    if chunk is not None:
        # down sampling
        chunk[:, -1] = np.abs(chunk[:, -1])
        chunk = downsampler.apply(chunk)

        # compute feedback score
        if len(chunk) > 0:
            # virtual_channel = chunk.dot(spatial_filter)
            # envelope = cfir.apply(virtual_channel)
            # envelope = artificial_delay.apply(envelope)
            # score = (envelope - mean)/(std if std > 0 else 1)

            # get mock chunk
            if current_sample + len(chunk) > len(mock_fb_signal):
                current_sample = 0
            score = mock_fb_signal[current_sample:current_sample + len(chunk)]
            current_sample += len(chunk)

            if play_feedback:
                # volume from mock score
                volume = (np.tanh(score[-1]) / 2 + 0.5)**3
                # print(score, volume)
                music.volume = volume
                # volume_controller.set_volume(volume)

            # push down-sampled chunk to lsl outlet
            chunk = np.hstack([chunk, score.reshape(-1, 1)])
            lsl_out.push_chunk(chunk.tolist())

    # handle NFBLab client messages
    meta_str, obj = server.pull_message()
    if meta_str is None:
        continue
    elif meta_str == 'msg':  # baseline blocks (set message and stop NFB)
        play_feedback = False
        music.volume = 0
        # volume_controller.set_volume(0)
        print('Dummy.. Set message to "{}"'.format(obj))
        message.text = obj
        if obj == 'Pause':
            voices['pause'].play()
        elif obj == 'Close':
            voices['close'].play()
        elif obj == 'Open':
            voices['open'].play()
        elif obj == 'Baseline':
            voices['baseline'].play()
        elif obj == 'Filters':
            voices['filters'].play()

        win.flip()
    elif meta_str == 'fb1':  # fb blocks
        voices['start'].play()
        play_feedback = True
        print('Dummy.. Run FB. Set message to "{}"'.format(obj))
        message.text = obj
        win.flip()
    elif meta_str == 'spt':  # update spatial filter
        spatial_filter = obj[:-1] # exclude FBSIGNAL channel
        print('Dummy.. Set spatial filter to {}'.format(obj))
    elif meta_str == 'bnd':  # update bandpass filter
        cfir = CFIRBandEnvelopeDetector(obj, FS_OUT, ExponentialSmoother(0.), n_taps=500)
        print('Dummy.. Set band to {}'.format(obj))
    elif meta_str == 'std':  # update fb score stats
        mean, std = obj
        print('Dummy.. Set stats to {}'.format(obj))



