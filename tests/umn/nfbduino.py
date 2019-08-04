import pylsl
from time import sleep
import numpy as np
import scipy.signal as sg
import serial


class CFIRBandEnvelopeDetector:
    def __init__(self, band, fs, delay, n_taps=500, n_fft=2000, weights=None, **kwargs):
        """
        Complex-valued FIR envelope detector based on analytic signal reconstruction
        :param band: freq. range to apply band-pass filtering
        :param fs: sampling frequency
        :param delay: delay of ideal filter in ms
        :param n_taps: length of FIR
        :param n_fft: length of freq. grid to estimate ideal freq. response
        :weights: least squares weights. If None match WHilbertFilter
        """
        w = np.arange(n_fft)
        H = 2 * np.exp(-2j * np.pi * w / n_fft * delay)
        H[(w / n_fft * fs < band[0]) | (w / n_fft * fs > band[1])] = 0
        F = np.array([np.exp(-2j * np.pi / n_fft * k * np.arange(n_taps)) for k in np.arange(n_fft)])
        if weights is None:
            self.b = F.T.conj().dot(H)/n_fft
        else:
            W = np.diag(weights)
            self.b = np.linalg.solve(F.T.dot(W.dot(F.conj())), (F.T.conj()).dot(W.dot(H)))
        self.a = np.array([1.])
        self.zi = np.zeros(len(self.b)-1)

    def apply(self, chunk: np.ndarray):
        y, self.zi = sg.lfilter(self.b, self.a, chunk, zi=self.zi)
        return y


class RectEnvDetector:
    def __init__(self, band, fs, delay, n_taps_bandpass, smooth_cutoff=None, **kwargs):
        """
        Envelope  detector  based  on  rectification  of  the  band-filtered  signal
        :param band: band of interest
        :param fs: sampling frequency
        :param n_taps_bandpass: FIR bandpass filter number of taps
        :param delay: desired delay to determine FIR low-pass filter number of taps
        :param smooth_cutoff: smooth filter cutoff frequency (if None equals to band length)
        """
        if n_taps_bandpass > 0:
            freq = [0, band[0], band[0], band[1], band[1], fs/2]
            gain = [0, 0, 1, 1, 0, 0]
            self.b_bandpass = sg.firwin2(n_taps_bandpass, freq, gain, fs=fs)
            self.zi_bandpass = np.zeros(n_taps_bandpass - 1)
        else:
            self.b_bandpass, self.zi_bandpass = np.array([1., 0]), np.zeros(1)

        if smooth_cutoff is None: smooth_cutoff = band[1] - band[0]

        n_taps_smooth = delay * 2 - n_taps_bandpass
        if n_taps_smooth > 0:
            self.b_smooth = sg.firwin2(n_taps_smooth, [0, smooth_cutoff, smooth_cutoff, fs/2], [1, 1, 0, 0], fs=fs)
            self.zi_smooth = np.zeros(n_taps_smooth - 1)
        elif n_taps_smooth == 0:
            self.b_smooth, self.zi_smooth = np.array([1., 0]), np.zeros(1)
        else:
            print('RectEnvDetector insufficient parameters: 2*delay < n_taps_bandpass. Filter will return nans')
            self.b_smooth, self.zi_smooth = np.array([np.nan, 0]), np.zeros(1)

    def apply(self, chunk):
        y, self.zi_bandpass = sg.lfilter(self.b_bandpass, [1.],  chunk, zi=self.zi_bandpass)
        y = np.abs(y)
        y, self.zi_smooth  = sg.lfilter(self.b_smooth, [1.], y, zi=self.zi_smooth)
        return y


if __name__ == '__main__':
    # settings
    IND_TYPE = 0
    IND_DURATION = 1
    IND_MESSAGE = 2
    IND_EVENT = 3
    CH_BLOCK = 0
    CH_SIGNAL = 1

    block_sequence = [('b', 20, 'seat steal', 'normalize'),
                      ('f', 60, 'feedback', '')]

    # connect to arduino
    ser = serial.Serial('COM7', 9600)
    sleep(3)
    ser.write(b'20r0g0b')

    # connect to EEG
    stream_info = pylsl.resolve_byprop('name', 'NVX136_Data')[0]
    fs = int(stream_info.nominal_srate())
    n_channels = stream_info.channel_count()
    stream = pylsl.StreamInlet(stream_info)

    # prepare variables
    n_blocks = len(block_sequence)
    n_all_samples = sum([block[IND_DURATION]*fs for block in block_sequence])

    record = np.empty((n_all_samples + fs*10, n_channels + 2))

    all_samples_counter = 0
    block_samples_counter = 0
    block_number = 0
    block_samples = block_sequence[block_number][IND_DURATION]*fs
    end_experiment = False

    # init envelope detector
    env_detector = CFIRBandEnvelopeDetector([8, 12], fs, int(100*fs/1000))
    env_max = 1

    # run experiment
    while not end_experiment:
        chunk = stream.pull_chunk()[0]
        # skip empty chunks
        if len(chunk) > 0:
            # save new data
            n_chunk_samples = len(chunk)
            all_samples_counter += n_chunk_samples
            block_samples_counter += n_chunk_samples
            record[all_samples_counter - n_chunk_samples:all_samples_counter, :n_channels] = chunk
            record[all_samples_counter - n_chunk_samples:all_samples_counter, n_channels + CH_BLOCK] = block_number
            env = np.abs(env_detector.apply(record[all_samples_counter - n_chunk_samples:all_samples_counter, 0]))
            record[all_samples_counter - n_chunk_samples:all_samples_counter, n_channels + CH_SIGNAL] = env

            # send envelope to LED
            if block_sequence[block_number][IND_TYPE] == 'f':
                ser.write('{}b'.format(int(record[all_samples_counter-1,-1]/env_max*30)).encode())

            # handle block sequence
            if block_samples_counter > block_samples:

                # update normalization coefficient
                if block_sequence[block_number][IND_EVENT] == 'normalize':
                    env_rec = record[all_samples_counter-block_samples_counter:all_samples_counter, n_channels+CH_SIGNAL]
                    env_max = np.percentile(env_rec, 95)

                # update block
                block_number += 1

                # end experiment if last block
                if block_number >= n_blocks:
                    record = record[:all_samples_counter]
                    ser.close()
                    end_experiment = True
                # switch blocks
                else:
                    block_samples = block_sequence[block_number][IND_DURATION]*fs
                    block_samples_counter = 0
                    ser.write(b'20r0g0b' if block_sequence[block_number][IND_TYPE] == 'b' else b'0r0g0b')
        # sleep to next chunk
        sleep(0.001)

    # save experiment data
    np.save('led_nfb.npy', record)

