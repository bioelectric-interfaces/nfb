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

from pynfb.protocols.ssd.topomap_selector_ica import ICADialog

settings_file = 'D:\\vnd_spbu\\ica\\ica\\vnd_spbu_5days.json'
with open(settings_file, 'r') as f:
    settings = loads(f.read())

dir_ = settings['dir']
subj = 0
day = 1
#for subj in range(4):
    #for day in range(5):
experiments = settings['subjects'][subj]
experiment = experiments[day]

def preproc(x, fs, rej=None):
    #x = dc_blocker(x)
    x = fft_filter(x, fs, band=(0, 45))
    if rej is not None:
        x = np.dot(x, rej)
    return x


reject = True
with h5py.File('{}\\{}\\{}'.format(settings['dir'], experiment, 'experiment_data.h5')) as f:
    fs, channels, p_names = get_info(f, settings['drop_channels'])
    spatial = -f['protocol20/signals_stats/left/spatial_filter'][:]
    spatial /= (np.dot(spatial, spatial)**0.5)
    print(list(spatial))
    #rejection, alpha, ica = load_rejections(f, reject_alpha=True)
    raw_before = OrderedDict()
    for j, name in enumerate(p_names):
        x = preproc(f['protocol{}/raw_data'.format(j + 1)][:], fs, None)
        raw_before = add_data(raw_before, name, x, j)

# make csp:
from PyQt4.QtGui import QApplication

ap = QApplication([])

# all_keys = ['1. Close', '2. Open', '3. Left', '4. Right', '5. Close', '6. Open', '7. Left', '8. Right', '9. Close', '10. Open', '11. Left', '12. Right']
# all_keys = ['1. Close', '5. Close', '9. Close', '2. Open', '6. Open', '10. Open','3. Left',  '7. Left', '11. Left','4. Right', '8. Right', '12. Right']
all_keys = ['2. Open', '6. Open', '10. Open', '3. Left', '7. Left', '11. Left']
raw_data = np.concatenate([raw_before[key] for key in all_keys])
#rej, spat = ICADialog.get_rejection(raw_data, channels, fs, mode='csp', states=None)[:2]
#_, spat = ICADialog.get_rejection(np.dot(raw_data, rej.val), channels, fs, mode='csp', states=None)[:2]

csp_spat = spatial
csp_spat = csp_spat/(np.dot(csp_spat, csp_spat)**0.5)#np.dot(spat, rej.val)

# plot raw data
ch_plot = ['C4', 'P4', 'ICA']#, 'Pz', 'Fp1']
fig1, axes = plt.subplots(len(ch_plot), ncols=1, sharex=True, sharey=False)
print(axes)

#find median
x_all = []
for name, x in raw_before.items():
    if name in ['14. Baseline']:
        x_all.append(np.abs(hilbert(fft_filter(x, fs, (4, 8)))))
x_median = np.mean(np.concatenate(x_all), 0)

x_all = []
for name, x in raw_before.items():
    if name in ['14. Baseline']:
        x_all.append(np.abs(hilbert(np.dot(fft_filter(x, fs, (4, 8)), spatial))))
x_f_median = np.mean(np.concatenate(x_all))

x_all = []
for name, x in raw_before.items():
    x_all.append(np.abs(hilbert(np.dot(fft_filter(x, fs, (4, 8)), csp_spat))))
x_csp_median = np.mean(np.concatenate(x_all))
print(x_median, x_f_median)





# plot raw
t = 0
cm = get_colors_f
for name, x in list(raw_before.items()):
    for j, ch in enumerate(ch_plot):
        time = np.arange(t, t + len(x)) / fs
        y = x[:, channels.index(ch)] if ch not in  ['ICA', 'CSP'] else (np.dot(x, spatial) if ch == 'ICA' else np.dot(x, csp_spat))
        x_plot = fft_filter(y, fs, band=(3, 45)) if ch != 'Fp1' else x[:, channels.index(ch)]
        axes[j].plot(time, x_plot, c=cm(name), alpha=1)
        x_plot = fft_filter(y, fs, band=(11, 14)) if ch != 'Fp1' else x[:, channels.index(ch)]
        x_plot = np.abs(hilbert(x_plot))
        threshold = 1.5*x_median[channels.index(ch)] if ch not in  ['ICA', 'CSP'] else 1.5*x_f_median if ch == 'ICA' else 1.5*x_csp_median
        sc = 15*x_plot.mean()
        axes[j].fill_between(time, (-(x_plot > threshold) - 1) * sc, (x_plot>threshold)*sc, facecolor=cm(name), alpha=0.6, linewidth=0)
        axes[j].set_ylabel(ch)
        axes[j].set_ylim(-1e-4, 1e-4)
    t += len(x)
axes[0].set_title('Day {}'.format(day+1))
#plt.legend(['Closed','Opened', 'Right', 'Left', 'Baseline', 'FB'])

# plot spectrum
ch_plot = ['P4', 'C4', 'ICA']
print(raw_before.keys())
fig2, axes = plt.subplots(len(ch_plot), ncols=1, sharex=True, sharey=False, figsize=(15,9))
y_max = [0.4e-10, 2.5, 10, 20, 20][subj]
for j, ch in enumerate(ch_plot):
    leg = []
    for jj, key in enumerate(['14. Baseline', '11. Left', '15. FB', '17. FB', '19. FB','21. FB','23. FB',
                              '27. Left'] + ([] if day == 0 else ['35. Baseline'])):
                              #'2. Left', '14. Left']):
        x = raw_before[key]
        style = '--' if key in ['26. Open', '27. Left', '35. Baseline'] else ''
        w = 2
        if 'FB' in key:
            style = ''
            w = jj-1
        y = x[:, channels.index(ch)] if ch not in ['ICA', 'CSP'] else (np.dot(x, spatial) if ch == 'ICA' else np.dot(x, csp_spat))
        f, Pxx = welch(y, fs, nperseg=2048,)
        axes[j].plot(f, Pxx, style,  c=cm(key), linewidth=w, alpha=0.8 if 'FB' in key else 1)
        x_plot = np.abs(hilbert(fft_filter(y, fs, band=(11, 14))))
        threshold = 1.5 * x_median[channels.index(ch)] if ch not in ['ICA', 'CSP'] else 1.5 * x_f_median if ch == 'ICA' else 1.5 * x_csp_median
        leg.append('{1}:  D={0:.0f}% of {2:.1f} min'.format(sum((x_plot > threshold))/(len(x_plot))*100, key, len(x_plot)/fs/60))


    axes[j].set_xlim(7, 14)
    axes[j].set_ylim(0, y_max)

    axes[j].set_ylabel(ch)
    axes[j].legend(leg, loc='upper left')
axes[0].set_title('Day {}'.format(day+1))
fig2.savefig('FBSpec_S{}_D{}'.format(subj, day + 1))

plt.show()
