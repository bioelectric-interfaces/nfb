import numpy as np
import pylab as plt
import h5py
import mne
from mne.viz import plot_topomap
from pynfb.widgets.helpers import ch_names_to_2d_pos
from scipy.signal import welch, hilbert
from pynfb.postprocessing.utils import get_info, add_data_simple, fft_filter, dc_blocker, load_rejections
from collections import OrderedDict
from IPython.display import clear_output

# load raw
from json import loads

from pynfb.protocols.ssd.topomap_selector_ica import ICADialog

settings_file = 'C:\_NFB\old_desctop\\kriv\\vnd_spbu_5days.json'
with open(settings_file, 'r', encoding="utf-8") as f:
    settings = loads(f.read())

dir_ = settings['dir']
subj = 0
experiments = settings['subjects'][subj]
def preproc(x, fs, rej=None):
    #x = dc_blocker(x)
    #x = fft_filter(x, fs, band=(0, 45))
    if rej is not None:
        x = np.dot(x, rej)
    return x


#cm = get_colors2()
# plot spectrum
state_plot = ['Close', 'Open', 'Left', 'Right']
fig2, axes = plt.subplots(len(state_plot), ncols=4, sharex=True, sharey=True)
import seaborn as sns
cm = sns.color_palette('Paired')


peak = 50
raw = {}
names = ['Outside (corner)', 'Outside (center)', 'Inside (opened door)', 'Inside (closed door)']

for j_experiment, experiment in enumerate(experiments[:]):
    #experiment = experiments[0]
    reject = False
    with h5py.File('{}\\{}\\{}'.format(settings['dir'], experiment, 'experiment_data.h5')) as f:
        fs, channels, p_names = get_info(f, settings['drop_channels'])
        rejection, alpha, ica = None, None, None#load_rejections(f, reject_alpha=True)
        odict = OrderedDict()
        for j, name in enumerate(p_names):
            x = preproc(f['protocol{}/raw_data'.format(j + 1)][:], fs, rejection if reject else None)
            odict = add_data_simple(odict, name, x)
        raw[names[j_experiment]] = odict


    for j, key in enumerate(state_plot):

        f, Pxx = welch(raw[names[j_experiment]][key], fs, nperseg=2048, axis=0)
        #axes[j].semilogy(f, Pxx, alpha=1, c=cm[j_experiment*2+3])
        ax = axes[j, j_experiment]
        a, b = plot_topomap(np.log10(Pxx[np.argmin(np.abs(f-peak)), :]), ch_names_to_2d_pos(channels), cmap='Reds',
                            axes=ax, show=False, vmax=-10.5, vmin=-13)
        if j_experiment == 0:
            ax.set_ylabel(key)
        if j == len(state_plot)-1:
            ax.set_xlabel(names[j_experiment])

        #axes[j].set_xlim(0, 250)
        #axes[j].set_ylim(1e-19, 5e-10)
            #x_plot = np.abs(hilbert(fft_filter(raw_before[key][:, channels.index(ch)], fs)))
            #leg.append('P={:.3f}, D={:.3f}s'.format(Pxx[(f > 9) & (f < 14)].mean(), sum((x_plot > 5)) / fs/2))
        #axes[j].legend(leg)


fig2.colorbar(a)
plt.show()
