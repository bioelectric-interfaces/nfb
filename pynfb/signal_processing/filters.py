import numpy as np
from scipy.signal import butter, lfilter, savgol_coeffs
from scipy.fftpack import rfft, irfft, fftfreq
from  scipy import fftpack

class BaseFilter:
    def apply(self, chunk: np.ndarray):
        '''
        :param chunk:
        :return:
        '''
        raise NotImplementedError


class SpatialFilter(BaseFilter):
    def __init__(self, filters, topographies=None):
        '''
        Transform n_samples * n_channels data to n_samples * n_components
        :param filters: (n_channels, n_components) array or (n_channels, ) vector
        :param topography: corresponded to filters topographies
        '''
        self.filters = filters
        self.topographies = topographies

    def apply(self, chunk: np.ndarray):
        return np.dot(chunk, self.filters)


class SpatialRejection(BaseFilter):
    def __init__(self, val, rank=1, type_str='unknown', topographies=None):
        """
        :param val: np.array
        :param args: np.array args
        :param rank: rank of rejection
        :param type_str: source name of rejection
        :param topographies:  np.array with dim = (n x rank). It contains rank topography vectors with dim = n
                              (usually number of channels). If it's None, nan array will be created.
        :param kwargs: np.array kwargs
        """
        self.val = np.array(val)
        self.type_str = type_str
        self.rank = rank
        if topographies is not None:
            topographies = np.array(topographies)
            if topographies.ndim == 1:
                topographies = topographies.reshape((topographies.shape[0], 1))
            assert topographies.shape == (self.val.shape[0], self.rank), \
                'Bad topographies shape {}. Expected {}.'.format(topographies.shape, (self.val.shape[0], self.rank))
            self.topographies = topographies
        else:
            self.topographies = np.nan * np.zeros((self.val.shape[0], self.rank))

    def apply(self, chunk: np.ndarray):
        return np.dot(chunk, self.val)


class ButterFilter(BaseFilter):
    def __init__(self, band, fs, n_channels, order=4):
        self.n_channels = n_channels
        low, high = band
        if low is None and high is None:
            raise ValueError('band should involve one or two not None values')
        elif low is None:
            self.b, self.a = butter(order, high/fs*2, btype='low')
        elif high is None:
            self.b, self.a = butter(order, low/fs*2, btype='high')
        else:
            self.b, self.a = butter(order, [low/fs*2, high/fs*2], btype='band')
        self.zi = np.zeros((max(len(self.b), len(self.a)) - 1, n_channels))

    def apply(self, chunk: np.ndarray):
        y, self.zi = lfilter(self.b, self.a, chunk, axis=0, zi=self.zi)
        return y

    def reset(self):
        self.zi = np.zeros((max(len(self.b), len(self.a)) - 1, self.n_channels))

class FilterSequence(BaseFilter):
    def __init__(self, filter_sequence):
        self.sequence = filter_sequence

    def apply(self, chunk: np.ndarray):
        for filter_ in self.sequence:
            chunk = filter_.apply(chunk)
        return chunk


class FilterStack(BaseFilter):
    def __init__(self, filter_stack):
        self.stack = filter_stack

    def apply(self, chunk: np.ndarray):
        result = [filter_.apply(chunk) for filter_ in self.stack]
        return np.hstack(result)


class InstantaneousVarianceFilter(BaseFilter):
    def __init__(self, n_channels, n_taps):
        self.a = [1]
        self.b = np.ones(n_taps)/n_taps
        self.zi = np.zeros((max(len(self.a), len(self.b)) - 1, n_channels))

    def apply(self, chunk: np.ndarray):
        y, self.zi = lfilter(self.b, self.a, chunk**2, axis=0, zi=self.zi)
        return y

class Coherence(BaseFilter):
    def __init__(self, n_taps, fs, band):
        self.buffer = np.zeros((n_taps, 2))
        self.n_taps = n_taps
        self.w = fftpack.fftfreq(n_taps, 1 / fs)
        self.band = band
        h = np.zeros(n_taps)
        if n_taps % 2 == 0:
            h[0] = h[n_taps // 2] = 1
            h[1:n_taps // 2] = 2
        else:
            h[0] = 1
            h[1:(n_taps + 1) // 2] = 2
        self.h = np.repeat(h, 2).reshape(self.n_taps, 2)

    def apply(self, chunk: np.ndarray):
        if len(chunk) <= self.n_taps:
            self.buffer[:-len(chunk)] = self.buffer[len(chunk):]
            self.buffer[-len(chunk):] = chunk
        else:
            self.buffer = chunk[-self.n_taps:]
        Xf = fftpack.fft(self.buffer, self.n_taps, axis=0)
        Xf[np.abs(self.w) < self.band[0]] = 0
        Xf[np.abs(self.w) > self.band[1]] = 0
        H = Xf * self.h / np.sqrt(len(Xf))
        coh = np.dot(H[:, 0], H[:, 1].conj()) / np.sqrt(np.abs(
            np.dot(H[:, 0], H[:, 0].conj()) * np.dot(H[:, 1], H[:, 1].conj())))
        return np.ones(len(chunk)) * np.abs(np.imag(coh))


class ExponentialSmoother(BaseFilter):
    def __init__(self, factor):
        self.a = [1, -factor]
        self.b = [1 - factor]
        self.zi = np.zeros((max(len(self.a), len(self.b)) - 1, ))

    def apply(self, chunk: np.ndarray):
        y, self.zi = lfilter(self.b, self.a, chunk, zi=self.zi)
        return y

class SGSmoother(BaseFilter):
    def __init__(self, n_samples, sg_order):
        self.savgol_weights = savgol_coeffs(n_samples, sg_order, pos=n_samples - 1)
        self.b, self.a = (self.savgol_weights, [1.])
        self.zi = np.zeros((n_samples - 1, ))

    def apply(self, chunk: np.ndarray):
        y, self.zi = lfilter(self.b, self.a, chunk, zi=self.zi)
        return y


class FFTBandEnvelopeDetector(BaseFilter):
    def __init__(self, band, fs, factor, n_samples):
        # bandpass filter settings
        self.buffer = np.zeros((n_samples,))
        self.n_samples = n_samples
        self.w = fftfreq(2 * n_samples, d=1. / fs * 2)
        self.band = (band[0] or 0, band[1] or fs/2)

        # asymmetric gaussian window
        p = round(2 * n_samples * 2 / 4)  # maximum
        eps = 0.0001  # bounds value
        power = 2  # power of x
        left_c = - np.log(eps) / (p ** power)
        right_c = - np.log(eps) / (2 * n_samples - 1 - p) ** power
        samples_window = np.concatenate([np.exp(-left_c * abs(np.arange(p) - p) ** power),
                                         np.exp(-right_c * abs(np.arange(p, 2 * n_samples) - p) ** power)])
        self.samples_window = samples_window

        # exponential smoothing
        self.exp_smoother = ExponentialSmoother(factor)

    def apply(self, chunk: np.ndarray):
        # update buffer
        chunk_size = chunk.shape[0]
        self.chunk_size = chunk_size
        if chunk_size <= self.n_samples:
            self.buffer[:-chunk_size] = self.buffer[chunk_size:]
            self.buffer[-chunk_size:] = chunk
        else:
            self.buffer = chunk[-self.n_samples:]

        # bandpass filter and amplitude
        f_signal = rfft(np.hstack((self.buffer, self.buffer[-1::-1])) * self.samples_window)
        cut_f_signal = f_signal.copy()
        cut_f_signal[(self.w < self.band[0]) | (self.w > self.band[1])] = 0  # TODO: in one row
        y = np.ones_like(chunk) * np.abs(cut_f_signal).mean()

        # smoothing
        y = self.exp_smoother.apply(y)
        return y

class ButterBandEnvelopeDetector(BaseFilter):
    def __init__(self, band, fs, factor, order=4):
        self.butter_filter = ButterFilter(band, fs, 1, order=order)
        self.exp_smoother = ExponentialSmoother(factor)

    def apply(self, chunk: np.ndarray):
        y = self.butter_filter.apply(chunk[:, None])[:, 0]
        y = np.abs(y)
        y = self.exp_smoother.apply(y)
        return y


class SGBandEnvelopeDetector(BaseFilter):
    def __init__(self, band, fs, factor, n_samples, sg_order):
        self.exp_smoother = ExponentialSmoother(factor)
        self.buffer = np.zeros((n_samples,))
        self.n_samples = n_samples
        self.band = band
        # step 1: demodulation
        main_fq = (self.band[0] + self.band[1]) / 2
        #print(source_freq, main_fq)
        self.modulation = np.exp(-2j * np.pi * np.arange(1000 * fs / main_fq) / fs * main_fq)
        self.n_modulation_samples = len(self.modulation)
        self.modulation_timer = len(self.modulation) - 1
        # step 2: iir
        self.iir_b, self.iir_a = butter(1, (self.band[1] - self.band[0]) / fs)
        self.zf = [0]
        # step 3: sav gol
        self.savgol_weights = savgol_coeffs(self.n_samples, sg_order, pos=self.n_samples - 1)
        self.sg_b, self.sg_a = (self.savgol_weights, [1.])
        self.sg_zf = np.zeros((n_samples - 1, ))

    def apply(self, chunk: np.ndarray):
        # update buffer
        chunk_size = chunk.shape[0]
        self.chunk_size = chunk_size
        if chunk_size <= self.n_samples:
            self.buffer[:-chunk_size] = self.buffer[chunk_size:]
            self.buffer[-chunk_size:] = chunk
        else:
            self.buffer = chunk[-self.n_samples:]

        # bandpass filter and amplitude
        self.modulation_timer += chunk_size
        starting_index = (self.modulation_timer - self.chunk_size) % self.n_modulation_samples
        ending_index = self.modulation_timer % self.n_modulation_samples
        if starting_index > ending_index:
            part1 = self.modulation[starting_index:]
            part2 = self.modulation[:ending_index]
            result = np.concatenate([part1, part2])
        else:
            result = self.modulation[starting_index:ending_index]
        x = result * chunk
        # print(np.concatenate([[self.iir_buffer], x]))
        y, self.zf = lfilter(self.iir_b, self.iir_a, x, zi=self.zf)
        y, self.sg_zf = lfilter(self.sg_b, self.sg_a, y, zi=self.sg_zf)
        #y = self.exp_smoother.apply(y)
        y = np.ones_like(chunk) * np.abs(2 * y)
        return y

if __name__ == '__main__':
    import pylab as plt

    print('TEST: Butter Filter')
    n_samples = 10000
    time = np.arange(n_samples)/250
    data = np.sin(10 * 2 * np.pi * time.repeat(2).reshape(n_samples, 2))
    noise = np.random.normal(size=(n_samples, 1))*0.1
    butter_filter = ButterFilter((9, None), fs=250, n_channels=1)
    #plt.plot(time, data)
    #plt.plot(time, butter_filter.apply(data))
    #plt.plot(np.vstack([butter_filter.apply(data[k*200:(k+1)*200]) for k in range(n_samples//200)]), '--')
    #butter_filter.reset()
    #plt.plot(np.vstack([butter_filter.apply(data[k * 8:(k + 1) * 8]) for k in range(n_samples // 8)]), '--')


    print('TEST: Exp Smoother')
    data = np.sin(10 * 2 * np.pi * time)

    butter_filter = ExponentialSmoother(0.9)
    #plt.plot(data)
    #data = data + noise
    #plt.plot(data)
    #plt.plot(np.vstack([butter_filter.apply(data[k * 200:(k + 1) * 200]) for k in range(n_samples // 200)]), '--')


    print('TEST: butter env det')
    data = data + noise[:, 0]
    data[n_samples//2:] *= 0.5
    print(data.shape)
    butter_filter = ButterBandEnvelopeDetector((9, 12), 250, 0.99, 1)
    plt.plot(data)
    res = np.hstack([butter_filter.apply(data[k * 8:(k + 1) * 8]) for k in range(n_samples // 8)])
    plt.plot(1.55*res)
    #plt.show()

    print('TEST: fft env det')
    #data = data + noise
    #data[n_samples//2:] *= 0.5
    butter_filter = FFTBandEnvelopeDetector((9, 12), 250, 0.96, 500)
    #plt.plot(data)
    plt.plot(1.45*np.hstack([butter_filter.apply(data[k * 8:(k + 1) * 8]) for k in range(n_samples // 8)]))
    # plt.show()

    print('TEST: sg env det')
    #data = data + noise
    #data[n_samples//2:] *= 0.5
    butter_filter = SGBandEnvelopeDetector((9, 12), 250, 0.9, 151, 2)
    #plt.plot(data)
    plt.plot(np.hstack([butter_filter.apply(data[k * 8:(k + 1) * 8]) for k in range(n_samples // 8)]))
    plt.show()