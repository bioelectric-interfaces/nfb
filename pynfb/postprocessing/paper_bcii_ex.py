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

with h5py.File(r'C:\Users\Nikolai\Desktop\desctop_backup\neurotlon_data\D4\motors_leftlegs_11-22_13-19-36\experiment_data.h5') as f:
    fs, channels, p_names = get_info(f, ['A1', 'A2', 'AUX'])
    channels = [ch.split('-')[0] for ch in channels]
    mock_ind = [f['protocol{}'.format(k + 1)].attrs['mock_previous'] for k in range(len(p_names))]
    data = [f['protocol{}/raw_data'.format(k + 1)][:] for k in range(len(p_names))]
    times = [np.arange(len(x))/fs for x in data]
    signal = [f['protocol{}/signals_data'.format(k + 1)][:] for k in range(len(p_names))]
    print([k for k in f['protocol{}/signals_stats'.format(p_names.index('Pause')+1)]])
    spat = f['protocol{}/signals_stats/Signal1/spatial_filter'.format(p_names.index('Pause')+1)][:]
    band = f['protocol{}/signals_stats/Signal1/bandpass'.format(p_names.index('Pause')+1 )][:]



#print(p_names)
montage = Montage(channels)


print(mock_ind)
df = pd.DataFrame(np.concatenate(data), columns=channels)
df['block_name'] = np.concatenate([[p]*len(d) for p, d in zip(p_names, data)])
df['block_number'] = np.concatenate([[j + 1]*len(d) for j, d in enumerate(data)])
df['alpha'] = np.dot(df[channels], spat)
df['alpha_env'] = df['alpha']*0
df['time'] = np.arange(len(df)) / fs
df['times'] = np.concatenate(times)
df['signal'] = np.concatenate(signal)[:, 0]



fig, axes = plt.subplots(1, 5)

eeg = fft_filter(df.loc[df['block_name'].isin(['Legs', 'Left']), channels], fs, band)
topo = np.dot(np.dot(eeg.T, eeg), spat)

axes[2].set_xlabel('Spat. filt.')
plot_topomap(spat, montage.get_pos(), axes=axes[2], show=False, contours=0)
axes[3].set_xlabel('Topography')
plot_topomap(topo, montage.get_pos(), axes=axes[3], show=False, contours=0)

axes[0].plot(df.loc[df['block_number']==1, 'time'], 1000000*fft_filter(df.loc[df['block_number']==1, 'alpha'], fs, (2, 100)))
axes[0].set_xlim(3, 4)
axes[0].set_xlabel('Time, s')
axes[0].set_ylabel('Voltage, $\mu V$')
#axes[2].plot(fft_filter(df.loc[df['block_number']==2, 'alpha'], fs, (1, 45)))

axes[1].plot(*welch(df.loc[df['block_name'].isin(['Left']), 'alpha']*1000000, fs, nperseg=fs*4))
axes[1].plot(*welch(df.loc[df['block_name'].isin(['Legs']), 'alpha']*1000000, fs, nperseg=fs*4))
axes[1].legend(['Rest', 'Legs'])
axes[1].set_xlim(2, 30)
axes[1].set_xlabel('Freq., Hz')
axes[1].set_ylabel('PSD, $\mu V^2/Hz$')
axes[1].vlines(band, [-10]*2, [10]*2, alpha=0.8)
axes[1].set_ylim(0, 3)
#plt.show()

for n in df.loc[df['block_name'].isin(['Left', 'Legs']), 'block_number'].unique():
    df.loc[df['block_number'] == n, 'alpha_env'] = 1000000*np.abs(hilbert(fft_filter(df.loc[df['block_number'] == n, 'alpha'], fs, band)))


sns.tsplot(df[df['block_name'].isin(['Left', 'Legs'])], 'times', 'block_number', 'block_name', 'alpha_env', ax=axes[4], ci='sd')
axes[4].legend(['Rest', 'Legs'])
axes[4].set_xlabel('Time, s')
axes[4].set_ylabel('Voltage, $\mu V$')
for ax, t in zip(axes, 'abcde'):
    ax.set_title(t)
plt.show()