import numpy as np
import seaborn as sns
from numpy.fft import fftfreq
from scipy.fftpack import rfft, irfft

from pynfb.io.xml_ import get_lsl_info_from_xml


def dc_blocker(x, r=0.99):
    # DC Blocker https://ccrma.stanford.edu/~jos/fp/DC_Blocker.html
    y = np.zeros_like(x)
    for n in range(1, x.shape[0]):
        y[n] = x[n] - x[n-1] + r * y[n-1]
    return y


def fft_filter(x, fs, band=(9, 14)):
    w = fftfreq(x.shape[0], d=1. / fs * 2)
    f_signal = rfft(x, axis=0)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(w < band[0]) | (w > band[1])] = 0
    cut_signal = irfft(cut_f_signal, axis=0)
    return cut_signal


def get_power2(x, fs, band, n_sec=5):
    n_steps = int(n_sec * fs)
    w = fftfreq(n_steps, d=1. / fs * 2)
    print(len(range(0, x.shape[0] - n_steps, n_steps)))
    pows = [2*np.sum(rfft(x[k:k+n_steps])[(w > band[0]) & (w < band[1])]**2)/n_steps
            for k in range(0, x.shape[0] - n_steps, n_steps)]
    return np.array(pows)


def get_power(x, fs, band):
    #w = 0.
    #taps = firwin2(1000, [0, band[0]-w, band[0], band[1], band[1]+w, fs/2], [0, 0, 1, 1, 0, 0], nyq=fs/2)
    #x = filtfilt(taps, [1.], x)
    x = fft_filter(x, fs, band)
    return x**2


def load_rejections(f, reject_alpha=True):
    rejections = [f['protocol1/signals_stats/left/rejections/rejection{}'.format(j + 1)][:] for j in range(2)]
    alpha = f['protocol1/signals_stats/left/rejections/rejection2_topographies'][:]
    ica = f['protocol1/signals_stats/left/rejections/rejection1_topographies'][:]
    rejection = rejections[0]
    if reject_alpha:
        rejection = np.dot(rejection, rejections[1])
    return rejection, alpha, ica


def get_info(f, drop_channels):
    labels, fs = get_lsl_info_from_xml(f['stream_info.xml'][0])
    print('fs: {}\nall labels {}: {}'.format(fs, len(labels), labels))
    channels = [label for label in labels if label not in drop_channels]
    print('selected channels {}: {}'.format(len(channels), channels))
    n_protocols = len([k for k in f.keys() if ('protocol' in k and k != 'protocol0')])
    protocol_names = [f['protocol{}'.format(j+1)].attrs['name'] for j in range(n_protocols)]
    print('protocol_names:', protocol_names)
    return fs, channels, protocol_names


def get_protocol_power(f, i_protocol, fs, rejection, ch, band=(9, 14), dc=False):
    raw = f['protocol{}/raw_data'.format(i_protocol + 1)][:]
    x = np.dot(raw, rejection)[:, ch]
    if dc:
        x = dc_blocker(x)
    return get_power2(x, fs, band), fft_filter(x, fs, band), x


def get_colors():
    p_names = [ 'Right', 'Left',  'Rest', 'FB', 'Closed', 'Opened', 'Baseline']
    cm = sns.color_palette('Paired', n_colors=len(p_names))
    c = dict(zip(p_names, [cm[j] for j in range(len(p_names))]))
    return c


def add_data(powers, name, pow, j):
    if name == 'Filters':
        powers['{}. Closed'.format(j + 1)] = pow[:len(pow) // 2]
        powers['{}. Opened'.format(j + 1)] = pow[len(pow) // 2:]
    elif name == 'Rotate':
        powers['{}. Right'.format(j + 1)] = pow[:len(pow) // 2]
        powers['{}. Left'.format(j + 1)] = pow[len(pow) // 2:]
    else:
        powers['{}. {}'.format(j + 1, name)] = pow
    return powers


def add_data_simple(powers, name, pow):
    if name == 'Filters':
        powers['Closed'] = pow[:len(pow) // 2]
        powers['Opened'] = pow[len(pow) // 2:]
    elif name == 'Rotate':
        powers['Right'] = pow[:len(pow) // 2]
        powers['Left'] = pow[len(pow) // 2:]
    else:
        powers['{}'.format(name)] = pow
    return powers