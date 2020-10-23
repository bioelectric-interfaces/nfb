import numpy as np
import pylab as plt
import pandas as pd
from collections import OrderedDict
from json import loads
from mne.viz import plot_topomap
from pynfb.postprocessing.utils import get_info, add_data, get_colors_f, fft_filter, dc_blocker, load_rejections, \
    find_lag
from pynfb.protocols.ssd.topomap_selector_ica import ICADialog
from scipy.signal import welch, hilbert
import h5py


def preproc(x, fs, rej=None):
    #x = dc_blocker(x)
    #x = fft_filter(x, fs, band=(0, 70))
    if rej is not None:
        x = np.dot(x, rej)
    return x

def compute_lengths(x, gap, minimum):
    lengths = []
    x_copy = -x.astype(int).copy()+1
    c = 0
    for j, y in enumerate(x_copy):
        if y:
            c += 1
        elif c > 0:
            if c > gap:
                lengths.append(c)
            else:
                x_copy[j-c:j] = 0
            c = 0
    if len(lengths) == 0:
        lengths = [0]
    if minimum is None:
        #print(np.array(lengths)/500)
        return np.array(lengths), x_copy
    else:
        return compute_lengths(x_copy.copy(), minimum, None)
# load raw
from pynfb.widgets.helpers import ch_names_to_2d_pos

mock = False


dir_ = r'D:\bci_nfb_bci\bci_nfb_bci'
with open(dir_ + '\\info.json', 'r', encoding="utf-8") as f:
    settings = loads(f.read())

subj = 0
day = 2
experiment = settings['subjects'][subj][day]
with h5py.File('{}\\{}\\{}'.format(dir_, experiment, 'experiment_data.h5')) as f:
    fs, channels, p_names = get_info(f, settings['drop_channels'])
    spatial = f['protocol15/signals_stats/left/spatial_filter'][:]
    mu_band = f['protocol15/signals_stats/left/bandpass'][:]
    max_gap = 1 / min(mu_band) * 2
    min_sate_duration = max_gap * 2
    raw = OrderedDict()
    signal = OrderedDict()
    for j, name in enumerate(p_names):
        if name != 'Bci':
            x = preproc(f['protocol{}/raw_data'.format(j + 1)][:], fs)
            raw = add_data(raw, name, x, j)
            signal = add_data(signal, name, f['protocol{}/signals_data'.format(j + 1)][:], j)
    ch_plot = ['C3', 'P3', 'ICA']  # , 'Pz', 'Fp1']
    fig1, axes = plt.subplots(len(ch_plot), ncols=1, sharex=True, sharey=False)
    # find median
    x_all = []
    x_all_ica = []
    for name, x in raw.items():
        if 'Baseline' in name:
            x_all.append(np.abs(hilbert(fft_filter(x, fs, band=mu_band))))
            x_all_ica.append(np.abs(hilbert(np.dot(fft_filter(x, fs, band=mu_band), spatial))))
            break
    x_median = np.median(np.concatenate(x_all), 0)
    x_f_median = np.median(np.concatenate(x_all_ica))

    coef = 1

    # plot raw
    t = 0
    cm = get_colors_f
    fff = plt.figure()
    for name, x in list(raw.items()):
        for j, ch in enumerate(ch_plot):
            time = np.arange(t, t + len(x)) / fs
            y = x[:, channels.index(ch)] if ch != 'ICA' else np.dot(x, spatial)
            x_plot = fft_filter(y, fs, band=(2, 45))
            axes[j].plot(time, x_plot, c=cm(name), alpha=1)
            #envelope = np.abs(hilbert(fft_filter(y, fs, band=mu_band)))

            # axes[j].plot(time, envelope, c=cm(name), alpha=1, linewidth=1)

            #axes[j].plot(time, signal[name][:, 0]* envelope.std() + envelope.mean(), c='k', alpha=1)
            #threshold = coef * x_median[channels.index(ch)] if ch != 'ICA' else coef * x_f_median
            #sc = envelope.mean()

            #lengths, x_copy = compute_lengths(envelope > threshold, fs * max_gap, fs * min_sate_duration)
            #axes[j].fill_between(time, -x_copy * sc*0, (envelope > threshold)*sc, facecolor=cm(name), alpha=0.6, linewidth=0)

            axes[j].set_ylabel(ch)
            axes[j].set_xlabel('time, [s]')
            # axes[j].set_ylim(-1e-4, 1e-4)
            # axes[j].set_xlim(190, 205)
        t += len(x)
    axes[0].set_xlim(0, t / fs)
    axes[0].set_title('Day {}'.format(day + 1))

print('af')
plt.show()