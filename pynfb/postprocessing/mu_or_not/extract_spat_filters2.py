from pynfb.postprocessing.mu_or_not.meta import *
from pynfb.signal_processing.decompositions import CSPDecomposition
from mne.viz import plot_topomap
from pynfb.inlets.montage import Montage

group = 'treatment'
subj = 'p6'
day = 1
mat = loadmat(r'C:\Users\Nikolai\Desktop\Liza_diplom_data\Liza_diplom_data\treatment\p4\bci\p4_11.mat')
channels = [ch[0] for ch in mat['chan_names'][0]]
montage = Montage(channels)
df = pd.DataFrame(data=mat['data_cur'].T, columns=channels)

print(channels)
print(1000*np.isin(np.array(channels), ['Fp1', 'Fp2', 'F7', 'F8', 'Ft9', 'Ft10', 'T7', 'T8', 'Tp9', 'Tp10']))

df['state'] = mat['states_cur'][0]
print(df['state'].unique())

df = df.loc[~get_outliers_mask(df[channels], iter_numb=10, std=3)]

plt.plot(df[channels])
plt.show()

ica = CSPDecomposition(channels, fs, band)
df = df.loc[df.state.isin([6, 1])]
scores, filters, topos = ica.decompose(df[channels].as_matrix(), df['state']-1)
for k in range(len(topos)):
    plot_topomap(topos[:, k], montage.get_pos())
sources = fft_filter(np.dot(df[channels], filters), fs, band)
desyncs = sources[df.state == 6].std(0)/sources[df.state == 1].std(0)
smr_ind = np.argmax(desyncs)
df['SMR'+str(1)] = np.dot(df[channels], filters[:, smr_ind])
plot_topomap(filters[:, smr_ind], montage.get_pos())
plot_topomap(topos[:, smr_ind], montage.get_pos())
#df['SMR-env'+str(state)] = np.abs(hilbert(fft_filter(df[state+'SMR'], fs, [9, 13])))
plt.plot(df['SMR'+str(1)])
plt.plot(df['C3'])
plt.plot(df['C4'])
plt.show()