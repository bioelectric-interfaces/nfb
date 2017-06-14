import numpy as np
from scipy.signal import butter, lfilter


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

if __name__ == '__main__':
    import pylab as plt

    print('TEST: Butter Filter')
    n_samples = 10000
    time = np.arange(n_samples)/250
    data = np.sin(10 * 2 * np.pi * time.repeat(2).reshape(n_samples, 2)) + np.random.normal(size=(n_samples, 1))*0.1
    butter_filter = ButterFilter((9, None), fs=250, n_channels=1)
    #plt.plot(time, data)
    #plt.plot(time, butter_filter.apply(data))
    plt.plot(np.vstack([butter_filter.apply(data[k*200:(k+1)*200]) for k in range(n_samples//200)]), '--')
    butter_filter.reset()
    plt.plot(np.vstack([butter_filter.apply(data[k * 8:(k + 1) * 8]) for k in range(n_samples // 8)]), '--')
    plt.show()
