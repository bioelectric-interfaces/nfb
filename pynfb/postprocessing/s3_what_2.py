import numpy as np
import pylab as plt
import h5py
import mne

from scipy.signal import welch, hilbert
from pynfb.postprocessing.utils import get_info, add_data, get_colors_f, fft_filter, dc_blocker, load_rejections
from collections import OrderedDict
from IPython.display import clear_output

# load raw
from json import loads

settings_file = 'D:\\vnd_spbu\\pilot\\mu5days\\vnd_spbu_5days.json'
with open(settings_file, 'r', encoding="utf-8") as f:
    settings = loads(f.read())

dir_ = settings['dir']
subj = 3
day = 2
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


reject = False
with h5py.File('{}\\{}\\{}'.format(settings['dir'], experiment, 'experiment_data.h5')) as f:
    fs, channels, p_names = get_info(f, settings['drop_channels'])
    rejection, alpha, ica = load_rejections(f, reject_alpha=True)
    raw_before = OrderedDict()
    for j, name in enumerate(p_names):
        x = preproc(f['protocol{}/raw_data'.format(j + 1)][:], fs, rejection if reject else None)
        raw_before = add_data(raw_before, name, x, j)



# plot raw data
ch_plot = ['C3', 'C4', 'P3', 'P4']#, 'Pz', 'Fp1']
fig1, axes = plt.subplots(len(ch_plot), ncols=1, sharex=True, sharey=True)
print(axes)

#find median
x_all = []
for name, x in raw_before.items():
    x_all.append(np.abs(hilbert(fft_filter(x, fs, (4, 8)))))
x_median = np.mean(np.concatenate(x_all), 0)
print(x_median)


# plot raw
t = 0
cm = get_colors_f
for name, x in list(raw_before.items()):
    for j, ch in enumerate(ch_plot):
        time = np.arange(t, t + len(x)) / fs
        x_plot = fft_filter(x[:, channels.index(ch)], fs, band=(3, 45)) if ch != 'Fp1' else x[:, channels.index(ch)]
        axes[j].plot(time, x_plot, c=cm(name), alpha=1)
        x_plot = fft_filter(x[:, channels.index(ch)], fs, band=(9, 14)) if ch != 'Fp1' else x[:, channels.index(ch)]
        x_plot = np.abs(hilbert(x_plot))
        threshold = 1.5*x_median[channels.index(ch)]
        axes[j].fill_between(time, (-(x_plot > threshold) - 1) * 30, (x_plot>threshold)*30, facecolor=cm(name), alpha=0.6, linewidth=0)
        axes[j].set_ylabel(ch)
    t += len(x)
plt.legend(['Closed','Opened', 'Right', 'Left', 'Baseline', 'FB'])


# plot spectrum
ch_plot = ['C3', 'C4']
print(raw_before.keys())
fig2, axes = plt.subplots(len(ch_plot), ncols=1, sharex=True, sharey=True, figsize=(15,9))
y_max = [5, 2.5, 10, 20, 20][subj]
for j, ch in enumerate(ch_plot):
    leg = []
    for jj, key in enumerate(['1. Closed', '1. Opened', '3. Baseline', '15. Baseline', '4. FB', '6. FB', '8. FB','10. FB','12. FB',]):
                              #'2. Left', '14. Left']):
        x = raw_before[key]
        style = '--' if key in ['15. Baseline', '14. Left'] else ''
        w = 2
        if 'FB' in key:
            style = ''
            w = jj-2
        f, Pxx = welch(x[:, channels.index(ch)], fs, nperseg=2048,)
        axes[j].plot(f, Pxx, style,  c=cm(key), linewidth=w, alpha=0.8 if 'FB' in key else 1)
        x_plot = np.abs(hilbert(fft_filter(x[:, channels.index(ch)], fs)))
        leg.append('{2}: P={0:.3f}, D={1:.0f}% of {3:.1f} min'.format(Pxx[(f>9) & (f<14)].mean(), sum((x_plot > 5))/(len(x_plot))*100, key, len(x_plot)/fs/60))


    axes[j].set_xlim(9, 14)
    axes[j].set_ylim(0, y_max)

    axes[j].set_ylabel(ch)
    axes[j].legend(leg, loc='upper left')
fig2.savefig('FBSpec_S{}_D{}'.format(subj, day + 1))

plt.show()
