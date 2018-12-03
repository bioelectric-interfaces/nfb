import numpy as np
import pylab as plt
import h5py
from scipy.signal import hilbert, firwin2, filtfilt
from scipy.fftpack import rfft, irfft, fftfreq

from ..serializers.xml_ import get_lsl_info_from_xml
from ..signals.rejections import Rejections
from ..signal_processing.filters import SpatialRejection


def dc_blocker(x, r=0.99):
    # DC Blocker https://ccrma.stanford.edu/~jos/fp/DC_Blocker.html
    y = np.zeros_like(x)
    for n in range(1, x.shape[0]):
        y[n] = x[n] - x[n-1] + r * y[n-1]
    return y

def get_power(x, fs, band):
    w = 0.
    taps = firwin2(2000, [0, band[0]-w, band[0], band[1], band[1]+w, fs/2], [0, 0, 1, 1, 0, 0], nyq=fs/2)
    x = filtfilt(taps, [1.], x)
    return x**2

def load_rejections(f, reject_alpha=True):
    rejections = [f['protocol1/signals_stats/left/rejections/rejection{}'.format(j + 1)][:] for j in range(2)]
    rejections_alpha_topo = f['protocol1/signals_stats/left/rejections/rejection2_topographies'][:]
    rejection = rejections[0]
    if reject_alpha:
        rejection = np.dot(rejection, rejections[1])
    return rejection, rejections_alpha_topo

def get_info(f, drop_channels):
    labels, fs = get_lsl_info_from_xml(f['stream_info.xml'][0])
    print('fs: {}\nall labels {}: {}'.format(fs, len(labels), labels))
    channels = [label for label in labels if label not in drop_channels]
    print('selected channels {}: {}'.format(len(channels), channels))
    return fs, channels

def get_protocol_power(f, i_protocol, fs, rejection, ch, band=(9, 14), dc=False):
    raw = f['protocol{}/raw_data'.format(i_protocol + 1)][:]
    x = np.dot(raw, rejection)[:, ch]
    if dc:
        x = dc_blocker(x)
    return get_power(x, fs, band)

if __name__ == '__main__':
    fs = 500
    band = (9, 14)
    x = np.random.normal(size=(5000, ))
    plt.plot(x, alpha=0.2)
    w = fftfreq(x.shape[0], d=1. / fs * 2)
    f_signal = rfft(x)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(w < band[0]) | (w > band[1])] = 0
    cut_signal = irfft(cut_f_signal)
    plt.plot(cut_signal)
    print(np.sum(np.abs(x)**2), 2*np.sum(np.abs(f_signal)**2)/x.shape[0])
    print(np.sum(np.abs(cut_signal) ** 2), 2 * np.sum(np.abs(cut_f_signal) ** 2)/x.shape[0])
    print(np.sum(np.abs(hilbert(x)**2))/2, np.sum(np.abs(hilbert(cut_signal)**2))/2)
    plt.figure()
    plt.plot(w, cut_f_signal**2)
    plt.show()
