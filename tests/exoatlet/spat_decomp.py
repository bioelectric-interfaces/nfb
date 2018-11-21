path_to_nfblab = r'C:\Projects\nfblab\nfb'
import sys
import numpy as np
import pylab as plt
import pandas as pd
import scipy.signal as sg
sys.path.insert(0, path_to_nfblab)
from utils.load_results import load_data
from pynfb.signal_processing.filters import ButterFilter
from pynfb.signal_processing.decompositions import ICADecomposition, CSPDecomposition
from pynfb.inlets.montage import Montage
from mne.viz import plot_topomap

# settings
h5_file = r'C:\Projects\nfblab\nfb\pynfb\results\exoatlet_kolai_stay_go_10-24_15-47-00\experiment_data.h5'
band = (15, 30)
method = 'ica'
np.random.seed(401)

# load data
df, fs, channels, p_names = load_data(h5_file)
fs = int(fs)
eeg_channels = channels[:30]
n_channels = len(eeg_channels)
montage = Montage(eeg_channels)
print('Fs: {}Hz\nAll channels: {}\nEEG channels: {}\nBlock sequence: {}'.format(
    fs, ', '.join(channels), ', '.join(eeg_channels), '-'.join(p_names)))

# pre filter
pre_filter = ButterFilter(band, fs, n_channels)
df[eeg_channels] = pre_filter.apply(df[eeg_channels])
df = df.iloc[fs*5:]

# spatial decomposition
if method == 'ica':
    decomposition = ICADecomposition(eeg_channels, fs)
elif method == 'csp':
    decomposition = CSPDecomposition(eeg_channels, fs)
else:
    raise ValueError('Bad method name. Use "ica" or "csp".')

# select data between first and second "pause" block
first_b_number = p_names.index('Pause') + 1
second_b_number  =  10000# p_names.index('Pause', 1) + 1
X = df.loc[(df.block_number>first_b_number) & (df.block_number<second_b_number)]

# fit decomposition
decomposition.fit(X[eeg_channels], X.block_name=='Go')

# init axes
n_rows = 5
n_cols = 6
fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(15, 10))
plt.subplots_adjust(hspace=1)

# sort by erd
erds = np.zeros(n_channels)
erd_band = band  # (18, 30)
for k in range(n_channels):
    filt = decomposition.filters[:, k]
    go_data = X.loc[X.block_name == 'Go', eeg_channels].values
    st_data = X.loc[X.block_name == 'Stay', eeg_channels].values
    freq, go_spec = sg.welch(go_data.dot(filt), fs)
    freq, st_spec = sg.welch(st_data.dot(filt), fs)
    freq_slice = (freq > erd_band[0]) & (freq < erd_band[1])
    erds[k] = (st_spec[freq_slice].mean() - go_spec[freq_slice].mean()) / st_spec[freq_slice].mean()

# plot axes
for j, k in enumerate(np.argsort(erds)[::-1]):
    topo = decomposition.topographies[:, k]
    filt = decomposition.filters[:, k]

    ax = axes[j // n_cols, j % n_cols * 2]
    plot_topomap(topo, montage.get_pos(), axes=ax, show=False, contours=0)
    ax.set_title(str(k))
    ax.set_xlabel('{:.1f}%'.format(erds[k] * 100))

    go_data = X.loc[X.block_name == 'Go', eeg_channels].values
    st_data = X.loc[X.block_name == 'Stay', eeg_channels].values
    freq, go_spec = sg.welch(go_data.dot(filt), fs)
    freq, st_spec = sg.welch(st_data.dot(filt), fs)
    freq_slice = (freq > 3) & (freq < 40)

    ax = axes[j // n_cols, j % n_cols * 2 + 1]
    ax.plot(freq[freq_slice], go_spec[freq_slice])
    ax.plot(freq[freq_slice], st_spec[freq_slice])
    ax.fill_between(freq[freq_slice], go_spec[freq_slice], st_spec[freq_slice], alpha=0.5)
    ax.get_yaxis().set_visible(False)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xticklabels([0, 10, 20, 30, 40])

plt.show()