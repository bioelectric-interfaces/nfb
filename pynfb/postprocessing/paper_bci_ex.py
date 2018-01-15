from pynfb.postprocessing.utils import get_info, fft_filter
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt
from pynfb.signal_processing.decompositions import ICADecomposition
from mne.viz import plot_topomap
from pynfb.inlets.montage import Montage
from scipy.signal import hilbert

with h5py.File(r'C:\Users\Nikolai\PycharmProjects\nfb\pynfb\results\bci-example_12-28_17-57-26\experiment_data.h5') as f:
    fs, channels, p_names = get_info(f, [])
    channels = [ch.split('-')[0] for ch in channels]
    data = [f['protocol{}/raw_data'.format(k + 1)][:] for k in range(len(p_names))]

print(p_names)
montage = Montage(channels)
df = pd.DataFrame(np.concatenate(data), columns=channels)
df['block_name'] = np.concatenate([[p]*len(d) for p, d in zip(p_names, data)])
df['block_number'] = np.concatenate([[j + 1]*len(d) for j, d in enumerate(data)])
df['times'] = np.concatenate([np.arange(len(d)) for d in data])
df = df[df['block_name'].isin(['Legs', 'Rest', 'Left', 'Right'])]

#
eeg = df[channels].as_matrix()

try:
    filters = np.load('filters_ex.npy')
    topos = np.load('topos_ex.npy')
except FileNotFoundError:
    ica = ICADecomposition(channels, fs, (0.5, 45))
    scores, filters, topos = ica.decompose(eeg)
    np.save('filters_ex.npy', filters)
    np.save('topos_ex.npy', topos)

sources = np.dot(eeg, filters)
for state in ['Legs', 'Left', 'Right']:
    smr_ind = np.argmax(sources[df.block_name == 'Rest'].std(0)/sources[df.block_name == state].std(0))
    df[state+'SMR'] = np.dot(df[channels], filters[:, smr_ind])

    df[state + 'SMR-env'] = np.abs(hilbert(fft_filter(df[state+'SMR'], fs, [9, 13])))
    #plt.plot(np.dot(df[channels], filters[:, smr_ind]))
    #plt.show()
    #plot_topomap(topos[:, smr_ind], montage.get_pos())
df.to_csv('wow_ex.csv')

sns.tsplot(df, time='times', unit='block_number', value='LegsSMR-env', condition='block_name')
plt.show()