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

with h5py.File(r'C:\Users\Nikolai\PycharmProjects\nfb\pynfb\results\alpha-nfb-example_01-13_17-37-02\experiment_data.h5') as f:
    fs, channels, p_names = get_info(f, ['A1', 'A2', 'AUX'])
    channels = [ch.split('-')[0] for ch in channels]
    mock_ind = [f['protocol{}'.format(k + 1)].attrs['mock_previous'] for k in range(len(p_names))]
    data = [f['protocol{}/raw_data'.format(k + 1)][:] for k in range(len(p_names))]
    signal = [f['protocol{}/signals_data'.format(k + 1)][:] for k in range(len(p_names))]
    spat = f['protocol{}/signals_stats/Alpha/spatial_filter'.format(p_names.index('Baseline')+1)][:]
    band = f['protocol{}/signals_stats/Alpha/bandpass'.format(p_names.index('Baseline') + 1)][:]



#print(p_names)
montage = Montage(channels)


print(mock_ind)
df = pd.DataFrame(np.concatenate(data), columns=channels)
df['block_name'] = np.concatenate([[p]*len(d) for p, d in zip(p_names, data)])
df['block_number'] = np.concatenate([[j + 1]*len(d) for j, d in enumerate(data)])
df['alpha'] = np.dot(df[channels], spat)
df['alpha_env'] = df['alpha']*0
df['time'] = np.arange(len(df)) / fs
df['signal'] = np.concatenate(signal)



fig, axes = plt.subplots(1, 5)

eeg = fft_filter(df.loc[df['block_name'].isin(['Open', 'Close']), channels], fs, band)
topo = np.dot(np.dot(eeg.T, eeg), spat)

axes[2].set_xlabel('Spat. filt.')
plot_topomap(spat, montage.get_pos(), axes=axes[2], show=False)
axes[3].set_xlabel('Topography')
plot_topomap(topo, montage.get_pos(), axes=axes[3], show=False)

axes[0].plot(df.loc[df['block_number']==2, 'time'], fft_filter(df.loc[df['block_number']==2, 'alpha'], fs, (2, 100)))
axes[0].set_xlim(31, 32)
axes[0].set_xlabel('Time, s')
axes[0].set_ylabel('Voltage, $\mu V$')
#axes[2].plot(fft_filter(df.loc[df['block_number']==2, 'alpha'], fs, (1, 45)))

axes[1].plot(*welch(df.loc[df['block_name'].isin(['Close']), 'alpha'], fs, nperseg=fs*4))
axes[1].plot(*welch(df.loc[df['block_name'].isin(['Open']), 'alpha'], fs, nperseg=fs*4))
axes[1].legend(['Close', 'Open'])
axes[1].set_xlim(5, 15)
axes[1].set_xlabel('Freq., Hz')
axes[1].set_ylabel('PSD, $\mu V^2/Hz$')
axes[1].vlines(band, [-50]*2, [50]*2, alpha=0.8)
axes[1].set_ylim(0, 40)
#plt.show()

for n in df.loc[df['block_name'].isin(['Real', 'Mock', 'Baseline']), 'block_number'].unique():
    df.loc[df['block_number'] == n, 'alpha_env'] = np.abs(hilbert(fft_filter(df.loc[df['block_number'] == n, 'alpha'], fs, band)))



df_fb = df[df['block_name'].isin(['Real', 'Mock'])]
print(df_fb['alpha_env'])
fig2, ax = plt.subplots(2, 1)
corr = [df_fb['signal'].corr(df_fb['alpha_env'].shift(i)) for i in range(100)]
print(corr)
ax[0].plot(corr)
ax[1].plot(df_fb['time'], df_fb['alpha_env'].shift(np.argmax(corr)))
ax[1].plot(df_fb['time'], df_fb['signal'])
ax[0].legend([str(np.argmax(corr)/fs) + ' ' + str(np.max(corr))])


coeff = df.loc[df['block_name'] == 'Baseline', 'alpha_env'].median()
#df['ma'] =
for n in df.loc[df['block_name'].isin(['Real', 'Mock', 'Baseline']), 'block_number'].unique():
    env = ((df.loc[df['block_number'] == n, 'alpha_env'] / coeff))**0.5
    roll = env.rolling(5, center=True)
    time = df.loc[df['block_number'] == n, 'time']
    #plt.plot(time, roll.median(), '--' if mock_ind[n-1] else '-', label=str(n)+p_names[n-1])
    reg = linregress(time, roll.median().fillna(method='ffill').fillna(method='bfill'))
    #plt.plot(time, time*reg.slope + reg.intercept)
    #plt.fill_between(time, roll.quantile(0.25), roll.quantile(0.75), alpha=0.8)
#plt.legend()
#plt.show()

n_slopes = 500
n_samples = fs*60
time = np.arange(n_samples)/fs
slopes = pd.DataFrame(columns=['condition', 'slope'])
for condition in ['Real', 'Mock']:
    blocks = df.loc[df['block_name'] == condition, 'block_number'].unique()
    for n in range(n_slopes):
        block = blocks[np.random.randint(0, len(blocks))]
        samples = df.loc[df['block_number']==block, 'alpha_env'].as_matrix()
        start = np.random.randint(0, len(samples)-n_samples)
        reg = linregress(time, samples[start:start+n_samples])
        slopes.loc[len(slopes)] = {'condition': condition, 'slope': reg.slope}

sns.boxplot(y='slope', x='condition', data=slopes, ax=axes[4])
#sns.swarmplot(y='slope', x='condition', data=slopes)
print(ttest_1samp(slopes.loc[slopes['condition']=='Real', 'slope'], 0))
print(ttest_1samp(slopes.loc[slopes['condition']=='Mock', 'slope'], 0))
print(ranksums(slopes.loc[slopes['condition']=='Real', 'slope'], slopes.loc[slopes['condition']=='Mock', 'slope']))
print(ttest_ind(slopes.loc[slopes['condition']=='Real', 'slope'], slopes.loc[slopes['condition']=='Mock', 'slope']))

for ax, t in zip(axes, 'abcde'):
    ax.set_title(t)
plt.show()