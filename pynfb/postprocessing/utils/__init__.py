import numpy as np
from scipy.signal import welch
from scipy import fftpack
import h5py
from pynfb.serializers.xml_ import get_lsl_info_from_xml
import pandas as pd
import pylab as plt


def dc_blocker(x, r=0.99):
    # DC Blocker https://ccrma.stanford.edu/~jos/fp/DC_Blocker.html
    y = np.zeros_like(x)
    for n in range(1, x.shape[0]):
        y[n] = x[n] - x[n-1] + r * y[n-1]
    return y


def fft_filter(x, fs, band=(9, 14)):
    w = fftpack.rfftfreq(x.shape[0], d=1. / fs)
    f_signal = fftpack.rfft(x, axis=0)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(w < band[0]) | (w > band[1])] = 0
    cut_signal = fftpack.irfft(cut_f_signal, axis=0)
    return cut_signal


def get_power2(x, fs, band, n_sec=5):
    n_steps = int(n_sec * fs)
    w = fftpack.fftfreq(n_steps, d=1. / fs * 2)
    print(len(range(0, x.shape[0] - n_steps, n_steps)))
    pows = [2*np.sum(fftpack.rfft(x[k:k+n_steps])[(w > band[0]) & (w < band[1])]**2)/n_steps
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
    import seaborn as sns
    p_names = [ 'Right', 'Left',  'Rest', 'FB', 'Close', 'Open', 'Baseline']
    cm = sns.color_palette('Paired', n_colors=len(p_names))
    c = dict(zip(p_names, [cm[j] for j in range(len(p_names))]))

    if 0:
        import pylab as plt
        import seaborn as sns
        sns.set_style('white')
        plt.figure()
        for name in p_names:
            plt.plot([0], [0], c=c[name])
        plt.legend(p_names)
        plt.savefig('legend.png', dpi=300)
        plt.show()

    return c

def get_colors_f(key):
    cm = get_colors()
    ind = [name for name in cm if name in key]
    return cm[ind[0]]

def get_colors2():
    p_names = [ 'Right', 'Left',  'Rest', 'FB', 'Close', 'Open', 'Baseline']

    cm = sns.color_palette('Paired', n_colors=len(p_names))

    c = dict(zip(p_names, [cm[j] for j in range(len(p_names))]))
    return c


def add_data(powers, name, pow, j):
    if name in ['Filters', 'Bci']:
        powers['{}. Rest'.format(j + 1)] = pow
    elif name == 'Rotate':
        powers['{}. Right'.format(j + 1)] = pow[:len(pow) // 2]
        powers['{}. Left'.format(j + 1)] = pow[len(pow) // 2:]
    elif name == 'Motor':
        powers['{}. Left'.format(j + 1)] = pow
    elif 'FB' in name:
        powers['{}. FB'.format(j + 1, name)] = pow[:]
    else:
        powers['{}. {}'.format(j + 1, name)] = pow
    return powers


def add_data_simple(odict, name, x):
    import mne
    to_raw = lambda y: y#mne.serializers.RawArray(y.T, info)
    if name == 'Filters':
        odict['Closed'] = to_raw(x[:len(x) // 2])
        odict['Opened'] = to_raw(x[len(x) // 2:])
    elif name == 'Rotate':
        odict['Right'] = to_raw(x[:len(x) // 2])
        odict['Left'] = to_raw(x[len(x) // 2:])
    else:
        odict[name] = to_raw(x)
    #elif 'FB' in name:
    #    odict['FB'] = to_raw(x)
    return odict

def find_lag(x, target, fs=None, show=False):
    n = 1000
    nor = lambda x:  (x - np.mean(x)) / np.std(x)
    lags = np.arange(n)
    mses = np.zeros_like(lags).astype(float)
    n_points = len(target) - n
    for lag in lags:
        mses[lag] = np.mean((nor(target[:n_points]) - nor(x[lag:n_points+lag]))**2)
    lag = np.argmin(mses)

    if show:
        import pylab as plt
        f, (ax1, ax2) = plt.subplots(2)
        ax1.plot(mses)
        ax1.plot(lag, np.min(mses), 'or')
        lag_str = '{}'.format(lag) if fs is None else '{} ({:.3f} s)'.format(lag, lag/fs)
        ax1.text(lag+n//100*2, np.min(mses), lag_str)
        ax2.plot(nor(target))
        ax2.plot(nor(x[lag:]), alpha=1)
        ax2.plot(nor(x), alpha=0.5)
        ax2.legend(['target',  'x[{}:]'.format(lag), 'x'])
        plt.show()
    return lag

def get_main_freq(x, fs, band_range=(8, 15), secperseg=4):
    f, pxx = welch(x, fs, nperseg=fs*secperseg)
    pxx[(f < band_range[0]) | (f > band_range[1])] = 0
    return f[np.argmax(pxx)]

def get_main_band(x, fs, band_range=(8, 15), band_width=2, secperseg=4):
    main_freq = get_main_freq(x, fs, band_range, secperseg)
    return [main_freq - band_width/2, main_freq + band_width/2]

def band_hilbert(x, fs, band, N=None, axis=-1):
    x = np.asarray(x)
    Xf = fftpack.fft(x, N, axis=axis)
    w = fftpack.fftfreq(x.shape[0], d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = fftpack.ifft(Xf, axis=axis)
    return 2*x


def load_data(file_path, drop_channels=()):
    with h5py.File(file_path) as f:
        fs, channels, p_names = get_info(f, drop_channels)
        data = [f['protocol{}/raw_data'.format(k + 1)][:] for k in range(len(p_names))]

        df = pd.DataFrame(np.concatenate(data), columns=channels)
        df['block_name'] = np.concatenate([[p]*len(d) for p, d in zip(p_names, data)])
        df['block_number'] = np.concatenate([[j + 1]*len(d) for j, d in enumerate(data)])

    return df, fs, p_names, channels

def load_signals_data(file_path, drop_channels=()):
    with h5py.File(file_path) as f:
        fs, channels, p_names = get_info(f, drop_channels)
        data = [f['protocol{}/signals_data'.format(k + 1)][:] for k in range(len(p_names))]
        df = pd.DataFrame(np.concatenate(data), columns=list(f['protocol0/signals_stats']))
    return df


def runica(x, fs, channels, mode='ica'):
    from PyQt5.QtWidgets import QApplication
    from pynfb.protocols.ssd.topomap_selector_ica import ICADialog
    a = QApplication([])
    ica = ICADialog(x, channels, fs, mode=mode)
    ica.exec_()
    a.exit()
    return ica.spatial, ica.topography

def runica2(x, fs, channels, names=('Right', 'Left'), mode='ica'):
    from PyQt5.QtWidgets import QApplication
    from pynfb.protocols.ssd.topomap_selector_ica import ICADialog
    a = QApplication([])
    res = []
    decomposition = None
    for n in names:
        print('*** Select component for condition: ' + n)
        ica = ICADialog(x, channels, fs, decomposition=decomposition, mode=mode)
        ica.exec_()
        res.append(np.array((ica.spatial, ica.topography)))
        decomposition = ica.decomposition
    a.exit()
    return res


if __name__ == '__main__':
    from mne.viz import plot_topomap
    from pynfb.inlets.montage import Montage
    from pynfb.generators import ch_names32
    montage = Montage(ch_names32)
    spatial, topo = runica(np.random.normal(size=(100000, 32)), 1000, montage.get_names(), mode='csp')
    plot_topomap(spatial, montage.get_pos())
    plot_topomap(topo, montage.get_pos())
