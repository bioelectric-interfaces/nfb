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

from pynfb.widgets.helpers import ch_names_to_2d_pos

settings_file = 'D:\\vnd_spbu\\pilot\\mu5days\\vnd_spbu_5days.json'
with open(settings_file, 'r', encoding="utf-8") as f:
    settings = loads(f.read())

dir_ = settings['dir']
subj = 0
day = 0


def preproc(x, fs, rej=None):
    x = dc_blocker(x)
    x = fft_filter(x, fs, band=(3, 45))
    if rej is not None:
        x = np.dot(x, rej)
    return x


from pynfb.protocols.ssd.topomap_selector_ica import ICADialog
from PyQt5 import QtGui, QtWidgets

a = QtWidgets.QApplication([])

fig1, axes1 = plt.subplots(ncols=3)

for subj in range(3,4):
    for day in range(2,5):
        experiments = settings['subjects'][subj]
        experiment = experiments[day]
        reject = True
        with h5py.File('{}\\{}\\{}'.format(settings['dir'], experiment, 'experiment_data.h5')) as f:
            fs, channels, p_names = get_info(f, settings['drop_channels'])
            rejection, alpha, ica = load_rejections(f, reject_alpha=False)
            raw_before = OrderedDict()
            raw_after = OrderedDict()
            for j, name in enumerate(p_names):
                if j < 3:
                    x = preproc(f['protocol{}/raw_data'.format(j + 1)][:], fs, rejection if reject else None)
                    raw_before = add_data_simple(raw_before, name, x)
                elif j > len(p_names) - 3:
                    x = preproc(f['protocol{}/raw_data'.format(j + 1)][:], fs, rejection if reject else None)
                    print(x.shape)
                    raw_after = add_data_simple(raw_after, name, x)



        xx = np.concatenate([raw_before['Opened'], raw_before['Baseline'], raw_after['Baseline'], raw_before['Left'], raw_before['Right'],  raw_after['Left'], raw_after['Right'], ])
        #xx[:, channels.index('C3')] = xx[:, channels.index('C3')]
        rejection, spatial, topography, unmixing_matrix, bandpass, _ = ICADialog.get_rejection(xx, channels, fs, mode='csp', states=None)
        from mne.viz import plot_topomap
        print(spatial)
        fig, axes = plt.subplots(ncols=rejection.topographies.shape[1])
        if not isinstance(axes, type(axes1)) :
            axes = [axes]
        for ax, top in zip(axes, rejection.topographies.T):
            plot_topomap(top, ch_names_to_2d_pos(channels), axes=ax, show=False)
        fig.savefig('csp_S{}_D{}.png'.format(subj, day+1))
        fig.show()



