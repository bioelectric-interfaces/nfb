import numpy as np
from scipy.signal import butter, lfilter


class BaseFilter:
    def apply(self, chunk):
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

    def apply(self, chunk):
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

    def apply(self, chunk):
        return np.dot(chunk, self.val)


class ButterFilter(BaseFilter):
    def __init__(self, band, fs, n_channels, order=4):
        low, high = band
        if low is None and high is None:
            raise ValueError('band should involve one or two not None values')
        elif low is None:
            self.b, self.a = butter(order, high/fs*2, btype='low')
        elif high is None:
            self.b, self.a = butter(order, low/fs*2, btype='high')
        else:
            self.b, self.a = butter(order, [low/fs*2, high/fs*2], btype='band')
        self.zi = np.zeros((max(len(self.a), len(self.a)) - 1, n_channels))

    def apply(self, chunk):
        y, self.zi = lfilter(self.b, self.a, chunk, axis=0, zi=self.zi)
        return y


if __name__ == '__main__':
    import pylab as plt

    print('TEST: Butter Filter')
    time = np.arange(10000)/250
    data = np.sin(10 * 2 * np.pi * time.repeat(2).reshape(10000, 2)) + np.random.normal(size=(10000, 2))*0.1
    butter_filter = ButterFilter((9, None), fs=250, n_channels=2)
    plt.plot(time, data)
    plt.plot(time, butter_filter.apply(data))
    plt.plot(time, np.vstack([butter_filter.apply(data[k*20:(k+1)*20]) for k in range(10000//20)]), '--')
    plt.show()
