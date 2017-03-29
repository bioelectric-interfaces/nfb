import numpy as np
import pylab as plt
import h5py
import mne
from pynfb.postprocessing.utils import get_info, add_data_simple, get_colors, fft_filter, dc_blocker
from collections import OrderedDict
from IPython.display import clear_output

# load raw
from json import loads

settings_file = 'C:\_data\mu5days\\vnd_spbu_5days.json'
with open(settings_file, 'r') as f:
    settings = loads(f.read())

dir_ = settings['dir']
subj = 3
experiments = settings['subjects'][subj]
experiment = experiments[0]

def preproc(x, fs):
    x = dc_blocker(x)
    x = fft_filter(x, fs, band=(0, 45))
    return x

with h5py.File('{}\\{}\\{}'.format(settings['dir'], experiment, 'experiment_data.h5')) as f:
    fs, channels, p_names = get_info(f, settings['drop_channels'])

    raw_before = OrderedDict()
    raw_after = OrderedDict()
    for j, name in enumerate(p_names):
        if j < 3:
            x = preproc(f['protocol{}/raw_data'.format(j + 1)][:], fs)
            raw_before = add_data_simple(raw_before, name, x)
        elif j > len(p_names) - 3:
            x = preproc(f['protocol{}/raw_data'.format(j + 1)][:], fs)
            print(x.shape)
            raw_after = add_data_simple(raw_after, name, x)



# plot raw data
ch_plot = ['C3', 'C4', 'P3', 'P4', 'Pz', 'Fp1']
fig, axes = plt.subplots(len(ch_plot), ncols=1, sharex=True, sharey=True)
print(axes)

t = 0
cm = get_colors()
leg = []
for name, x in [(k, v) for k, v in raw_before.items()] + [(k, v) for k, v in raw_after.items()]:
    print(name, x.shape)
    for j, ch in enumerate(ch_plot):
        print(np.arange(t, t+len(x)).shape, x[:, channels.index(ch)].shape)
        if ch != 'Fp1':
            axes[j].plot(np.arange(t, t+len(x))/fs, fft_filter(x[:, channels.index(ch)], fs, band=(3, 45)), c=cm[name])
        else:
            axes[j].plot(np.arange(t, t + len(x)) / fs, x[:, channels.index(ch)],       c=cm[name])
        axes[j].set_ylabel(ch)
    t += len(x)
plt.legend([name for name in raw_before])
plt.show()
