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
import pandas as pd
import seaborn as sns
from scipy import stats
from pynfb.signals.bci import BCISignal

dir_ = r'D:\bci_nfb_bci\bci_nfb_bci'
with open(dir_ + '\\info.json', 'r', encoding="utf-8") as f:
    settings = loads(f.read())


day = 0
subj = 0

experiment = settings['subjects'][subj][day]
with h5py.File('{}\\{}\\{}'.format(dir_, experiment, 'experiment_data.h5')) as f:
    fs, channels, p_names = get_info(f, settings['drop_channels'])
    spatial = f['protocol15/signals_stats/left/spatial_filter'][:]
    data = np.vstack([f['protocol{}/raw_data'.format(k+1)][:] for k in range(9)])

from pynfb.signal_processing.decompositions import ArtifactRejector


ica_art = ArtifactRejector(channels, fs)
ica_art.fit(data)

plt.plot(data[:, 2])
plt.plot(ica_art.apply(data)[:, 2])
plt.show()