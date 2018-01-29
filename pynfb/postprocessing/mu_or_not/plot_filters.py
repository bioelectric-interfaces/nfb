from pynfb.postprocessing.utils import get_info, fft_filter
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt
from pynfb.signal_processing.decompositions import ICADecomposition
from mne.viz import plot_topomap
from pynfb.inlets.montage import Montage
from scipy.signal import hilbert, welch
from scipy.stats import linregress, ttest_ind, ranksums, ttest_1samp
from pynfb.postprocessing.mu_or_not.meta import *


good_subjects = ['p8', 'p10', 'p13', 'VV']
norm_subjects = ['p4', 'p6', 'IO', 'KM']

all_subjects = good_subjects + norm_subjects

subj = all_subjects[0]
day = 1






fig, axes = plt.subplots(2*len(norm_subjects), 6)
for s, subj in enumerate(norm_subjects[:]):
    for day in range(2):
        axes[day + 2*s, 0].set_ylabel('{}-d{}'.format(subj, day+1))
        mat = loadmat(
            r'C:\Users\Nikolai\Desktop\Liza_diplom_data\Liza_diplom_data\treatment\{0}\bci\{0}_{1}1.mat'.format(subj, day+1))
        channels = [ch[0] for ch in mat['chan_names'][0]]
        montage = Montage(channels)
        df = pd.DataFrame(data=mat['data_cur'].T, columns=channels)
        df['state'] = mat['states_cur'][0]
        df = df.loc[~get_outliers_mask(df[channels], iter_numb=10, std=3)]
        df['block_name'] = df['state'].apply(lambda x: {6: 'Rest', 1: 'Left', 2: 'Right'}[x])
        filters = np.load(r'treatment\{0}d{1}_filters.npy'.format(subj, day+1))
        topos = np.load(r'treatment\{0}d{1}_topographies.npy'.format(subj, day+1))
        ind = np.load(r'treatment\{0}d{1}_smr_ind.npy'.format(subj, day+1))
        for k in range(2):
            spat = filters[:, ind[k]]
            topo = topos[:, ind[k]]
            df['smr'] = np.dot(df[channels], spat)

            axes[day + 2*s, 0 + 3*k].set_xlabel('Spat. filt.')
            plot_topomap(spat, montage.get_pos(), axes=axes[day + 2*s, 0+ 3*k], show=False, contours=0)
            axes[day + 2*s, 1+ 3*k].set_xlabel('Topography')
            plot_topomap(topo, montage.get_pos(), axes=axes[day + 2*s, 1+ 3*k], show=False, contours=0)


            axes[day + 2*s, 2+ 3*k].plot(*welch(df.loc[df['block_name'].isin(['Rest']), 'smr']*1000000, fs, nperseg=fs*4))
            axes[day + 2*s, 2+ 3*k].plot(*welch(df.loc[df['block_name'].isin(['Left']), 'smr']*1000000, fs, nperseg=fs*4))
            axes[day + 2*s, 2+ 3*k].plot(*welch(df.loc[df['block_name'].isin(['Right']), 'smr']*1000000, fs, nperseg=fs*4))

            if day == 0 and s == 0 and k ==1:
                axes[day + 2*s, 2+ 3*k].legend(['Rest', 'Left', 'Right'])
            axes[day + 2*s, 2+ 3*k].set_xlim(2, 30)
            axes[day + 2*s, 2+ 3*k].set_xlabel('Freq., Hz')
            axes[day + 2*s, 2+ 3*k].set_ylabel('PSD, $\mu V^2/Hz$')

plt.show()
