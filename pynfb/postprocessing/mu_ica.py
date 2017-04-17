import h5py
import numpy as np
import pylab as plt

from collections import OrderedDict
from json import loads
from mne.viz import plot_topomap
from pynfb.postprocessing.utils import get_info, add_data, get_colors_f, fft_filter, dc_blocker, load_rejections
from pynfb.protocols.ssd.topomap_selector_ica import ICADialog
from scipy.signal import welch, hilbert


# load raw
mock = False


settings_file = 'D:\\vnd_spbu\\ica\\ica\\vnd_spbu_5days.json' if not mock else 'D:\\vnd_spbu\\mock\\Pilot_mok\\vnd_spbu_5days.json'
with open(settings_file, 'r') as f:
    settings = loads(f.read())

dir_ = settings['dir']
subj = 1
day = 2
mu_band = (11.8, 14.8)
max_gap = 1 / min(mu_band) * 2
min_sate_duration = max_gap * 2

run_ica = mock
reject = False
#for subj in range(4):
    #for day in range(5):
experiments = settings['subjects'][subj]
experiment = experiments[day]

def preproc(x, fs, rej=None):
    x = dc_blocker(x)
    x = fft_filter(x, fs, band=(0, 45))
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

def compute_heights(x, mask):
    lengths = []
    mask_copy = mask.astype(int).copy()
    lengths_buffer = []
    for j, y in enumerate(mask_copy):
        if y:
            lengths_buffer.append(x[j])
        elif len(lengths_buffer) > 0:
            lengths.append(np.mean(lengths_buffer))
            lengths_buffer = []
    print(lengths)
    return np.array(lengths)

with h5py.File('{}\\{}\\{}'.format(settings['dir'], experiment, 'experiment_data.h5')) as f:
    fs, channels, p_names = get_info(f, settings['drop_channels'])
    if reject:
        rejections = load_rejections(f, reject_alpha=True)[0]
    else:
        rejections = None
    spatial = -f['protocol15/signals_stats/left/spatial_filter'][:]
    raw_before = OrderedDict()
    for j, name in enumerate(p_names):
        x = preproc(f['protocol{}/raw_data'.format(j + 1)][:], fs, rejections)
        raw_before = add_data(raw_before, name, x, j)

# make csp:
if run_ica:
    from PyQt4.QtGui import QApplication
    ap = QApplication([])
    all_keys = [key for key in raw_before.keys() if 'Left' in key or 'Right' in key or 'Close' in key or 'Open' in key]
    print(all_keys)
    raw_data = np.concatenate([raw_before[key] for key in all_keys])
    rej, spatial = ICADialog.get_rejection(raw_data, channels, fs, mode='ica', states=None)[:2]

# plot raw data
ch_plot = ['C3', 'P3', 'ICA']#, 'Pz', 'Fp1']
fig1, axes = plt.subplots(len(ch_plot), ncols=1, sharex=True, sharey=False)
print(axes)

#find median
x_all = []
x_all_ica = []
for name, x in raw_before.items():
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
for name, x in list(raw_before.items()):
    for j, ch in enumerate(ch_plot):
        time = np.arange(t, t + len(x)) / fs
        y = x[:, channels.index(ch)] if ch != 'ICA' else np.dot(x, spatial)
        x_plot = fft_filter(y, fs, band=(3, 45))
        axes[j].plot(time, x_plot, c=cm(name), alpha=1)
        envelope = np.abs(hilbert(fft_filter(y, fs, band=mu_band)))

        axes[j].plot(time, envelope, c=cm(name), alpha=1, linewidth=1)
        threshold = coef*x_median[channels.index(ch)] if ch != 'ICA' else coef*x_f_median
        sc = 15*envelope.mean()

        lengths, x_copy = compute_lengths(envelope > threshold, fs*max_gap, fs*min_sate_duration)
        print(all((envelope > threshold).astype(int) == x_copy))

        axes[j].fill_between(time, -x_copy * sc*0, (envelope > threshold)*sc, facecolor=cm(name), alpha=0.6, linewidth=0)
        axes[j].fill_between(time, -x_copy * sc*0, -(x_copy)*sc, facecolor=cm(name), alpha=0.8, linewidth=0)
        axes[j].set_ylabel(ch)
        #axes[j].set_ylim(-1e-4, 1e-4)
        #axes[j].set_xlim(432, 444)
    t += len(x)
axes[0].set_title('Day {}'.format(day+1))

keys = [key for key in raw_before.keys() if 'FB' in key or 'Baseline' in key]
# plot spectrum
ch_plot = ['P3', 'C4', 'ICA']
fig2, axes = plt.subplots(len(ch_plot), ncols=1, sharex=True, sharey=False, figsize=(15,9))
y_max = [0.4e-10, 2.5, 10, 20, 20][subj]
for j, ch in enumerate(ch_plot):
    leg = []
    fb_counter = 0
    for jj, key in enumerate(keys):
        x = raw_before[key]
        style = '--' if fb_counter > 0 else ''
        w = 2
        if 'FB' in key:
            fb_counter +=1
            style = ''
            w = fb_counter
        y = x[:, channels.index(ch)] if ch != 'ICA' else np.dot(x, spatial)
        f, Pxx = welch(y, fs, nperseg=2048,)
        axes[j].plot(f, Pxx, style,  c=cm(key), linewidth=w, alpha=0.8 if 'FB' in key else 1)
        x_plot = np.abs(hilbert(fft_filter(y, fs, band=mu_band)))
        threshold = coef * x_median[channels.index(ch)] if ch != 'ICA' else coef * x_f_median
        leg.append('{}'.format(key))


    axes[j].set_xlim(7, 14)
    #axes[j].set_ylim(0, 1e-10)
    axes[j].set_ylabel(ch)
    axes[j].legend(leg, loc='upper left')
axes[0].set_title('Day {}'.format(day+1))
fig2.savefig('FBSpec_S{}_D{}'.format(subj, day + 1))


# plot durations
fig3, axes = plt.subplots(nrows=5, sharex=True)
keys = ['14. Baseline', '17. FB', '19. FB', '21. FB', '23. FB', '25. FB', '27. Baseline']
keys = [key for key in raw_before.keys() if 'FB' in key or 'Baseline' in key]
import pandas as pd
dots = np.zeros((5, len(keys)))
for jj, key in enumerate(keys):
    y = np.dot(raw_before[key], spatial)
    f, Pxx = welch(y, fs, nperseg=2048, )
    envelope = np.abs(hilbert(fft_filter(y, fs, mu_band)))
    threshold = coef * x_f_median
    lengths, x_copy = compute_lengths(envelope > threshold, fs*max_gap, fs*min_sate_duration)
    heights = compute_heights(envelope, x_copy)
    dots[0, jj] = envelope.mean()
    dots[1, jj] = sum(x_copy) / (len(envelope)) * 100
    dots[2, jj] = len(lengths)/(len(envelope)/fs/60)
    dots[3, jj] = lengths.mean()/fs
    dots[4, jj] = heights.mean()
    for k in range(5):
        axes[k].plot([jj + 1], dots[k, jj], 'o', c=cm(key))

import seaborn as sns
for k in range(5):
    sns.regplot(np.arange(1, len(keys)+1), dots[k], ax=axes[k], color=cm('Baseline'),
                line_kws={'alpha':0.4}, ci=None)
    sns.regplot(np.arange(1, len(keys)+1), dots[k], ax=axes[k], color=cm('FB'),
                line_kws={'alpha':0.4}, ci=None)
    sns.regplot(np.arange(2, 7), dots[k,1:6], ax=axes[k], color=cm('FB'),
                line_kws={'alpha': 0.7}, ci=None, truncate=True)
    if len(keys) == 7:
        axes[k].plot([1, 7], dots[k,[0,6]], c=cm('Baseline'), alpha=0.7, linewidth=3)

titles = ['Mean envelope', 'Time in % for all mu-states', 'Number of mu-states per minute', 'Mean mu-state length [s]',
          'Mean mu-state envelope']
for ax, title in zip(axes, titles):
    ax.set_title(title)
    ax.set_xlim(0, len(keys)+1)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
    ax.set_xticks(range(1, len(keys)+1))
    ax.set_xticklabels(keys)
plt.suptitle('S{} Day{} Band: {}-{} Hz'.format(subj, day+1, *mu_band))
plt.savefig('S{}_Day{}'.format(subj, day+1))
plt.show()
