import numpy as np
import pylab as plt
import h5py
import mne

from scipy.signal import welch, hilbert
from pynfb.postprocessing.utils import get_info, add_data_simple, get_colors, fft_filter, dc_blocker, load_rejections
from collections import OrderedDict
from IPython.display import clear_output

# load raw
from json import loads

settings_file = 'C:\_data\mu5days\\vnd_spbu_5days.json'
with open(settings_file, 'r', encoding="utf-8") as f:
    settings = loads(f.read())

dir_ = settings['dir']
subj = 3
experiments = settings['subjects'][subj]
experiment = experiments[2]

def preproc(x, fs, rej=None):
    x = dc_blocker(x)
    x = fft_filter(x, fs, band=(0, 45))
    if rej is not None:
        x = np.dot(x, rej)
    return x


reject = False
with h5py.File('{}\\{}\\{}'.format(settings['dir'], experiment, 'experiment_data.h5')) as f:
    fs, channels, p_names = get_info(f, settings['drop_channels'])
    rejection, alpha, ica = load_rejections(f, reject_alpha=True)
    raw_before = OrderedDict()
    raw_after = OrderedDict()
    for j, name in enumerate(p_names):
        if j < 5:
            x = preproc(f['protocol{}/raw_data'.format(j + 1)][:], fs, rejection if reject else None)
            raw_before = add_data_simple(raw_before, name, x)
        elif j > len(p_names) - 3:
            x = preproc(f['protocol{}/raw_data'.format(j + 1)][:], fs, rejection if reject else None)
            print(x.shape)
            raw_after = add_data_simple(raw_after, name, x)



# plot raw data
ch_plot = ['C3', 'C4', 'P3', 'P4']#, 'Pz', 'Fp1']
fig1, axes = plt.subplots(len(ch_plot), ncols=1, sharex=True, sharey=True)
print(axes)

#find median
x_all = []
for name, x in list(raw_before.items()) + list(raw_after.items()):
    x_all.append(np.abs(hilbert(fft_filter(x, fs, (4, 8)))))
x_median = np.mean(np.concatenate(x_all), 0)
print(x_median)


# plot raw
t = 0
cm = get_colors()
for name, x in list(raw_before.items()) + list(raw_after.items()):
    if name in ['Closed', 'Opened', 'Baseline', 'Left', 'Right', 'FB']:
        for j, ch in enumerate(ch_plot):
            time = np.arange(t, t + len(x)) / fs
            x_plot = fft_filter(x[:, channels.index(ch)], fs, band=(3, 45)) if ch != 'Fp1' else x[:, channels.index(ch)]
            axes[j].plot(time, x_plot, c=cm[name], alpha=1)
            x_plot = fft_filter(x[:, channels.index(ch)], fs, band=(9, 14)) if ch != 'Fp1' else x[:, channels.index(ch)]
            x_plot = np.abs(hilbert(x_plot))
            threshold = 1.5*x_median[channels.index(ch)]
            axes[j].fill_between(time, (-(x_plot > threshold) - 1) * 30, (x_plot>threshold)*30, facecolor=cm[name], alpha=0.6, linewidth=0)
            axes[j].set_ylabel(ch)
        t += len(x)
plt.legend(['Closed','Opened', 'Right', 'Left', 'Baseline'])

# plot spectrum
ch_plot = ['C3', 'C4']

fig2, axes = plt.subplots(len(ch_plot), ncols=1, sharex=True, sharey=True)
for j, ch in enumerate(ch_plot):
    leg = []
    for key in ['Baseline', 'Left']:
        for x, style in zip([raw_before[key], raw_after[key]], ['', '--']):
            f, Pxx = welch(x[:, channels.index(ch)], fs, nperseg=2048,)
            axes[j].plot(f, Pxx, style,  c=cm[key])
            x_plot = np.abs(hilbert(fft_filter(x[:, channels.index(ch)], fs)))
            leg.append('P={:.3f}, D={:.2f}s'.format(Pxx[(f>9) & (f<14)].mean(), sum((x_plot > 5))/fs))

    f, Pxx = welch(raw_before['Opened'][:, channels.index(ch)], fs, nperseg=2048, )
    axes[j].plot(f, Pxx, c=cm['Opened'])
    axes[j].set_xlim(9, 14)
    x_plot = np.abs(hilbert(fft_filter(raw_before['Opened'][:, channels.index(ch)], fs)))
    leg.append('P={:.3f}, D={:.3f}s'.format(Pxx[(f > 9) & (f < 14)].mean(), sum((x_plot > 5)) / fs/2))
    axes[j].legend(leg)
    axes[j].set_ylabel(ch)

plt.show()
