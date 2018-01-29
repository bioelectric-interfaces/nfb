from pynfb.postprocessing.mu_or_not.meta import *
montage = Montage(channels)
subjects = ['p8', 'p10', 'p13', 'p6', 'VV']
subj = subjects[4]
subj = 'KM'


k = 1
day = 2

# load filters
filters = np.load(r'treatment\{0}d{1}_filters.npy'.format(subj, day))
topos = np.load(r'treatment\{0}d{1}_topographies.npy'.format(subj, day))
ind = np.load(r'treatment\{0}d{1}_smr_ind.npy'.format(subj, day))
ch = [c for c in channels if c != 'Pz']
left = filters[:,ind[np.argmax(np.abs(topos[ch.index('C4'), ind]))]]
right = filters[:,ind[np.argmax(np.abs(topos[ch.index('C3'), ind]))]]
from mne.viz import plot_topomap
plot_topomap(left, Montage(ch).get_pos())
plot_topomap(right, Montage(ch).get_pos())

baseline = loadmat(r'{0}\treatment\{1}\day{3}\baseline\baseline{2}.mat'.format(data_dir, subj, 1, day))['X1Topo'].T
baseline = baseline[~get_outliers_mask(baseline, iter_numb=20, std=3)]

data = [baseline]
sessions = [[0]*len(data[-1])]
for k in range(15):
    mat = loadmat(r'{0}\treatment\{1}\day{3}\proba{2}.mat'.format(data_dir, subj, k+1, day))['X1Topo'].T
    clear = mat[~get_outliers_mask(mat, iter_numb=20, std=3)]
    data.append(clear)
    sessions.append([k+1]*len(data[-1]))
df = pd.DataFrame(columns=channels, data=np.concatenate(data))
df['session'] = np.concatenate(sessions)

for ch in['Pz']:
    channels.remove(ch)
    del df[ch]

df['C3env'] = df['C3']*0
df['C4env'] = df['C4']*0
df['SMRLeftEnv'] = df['C3']*0
df['SMRRightEnv'] = df['C3']*0
for s in df['session'].unique():
    print(s)
    df.loc[df['session'] == s, 'C3env'] = np.abs(hilbert(fft_filter(df.loc[df['session'] == s, 'C3'], fs, band)))
    df.loc[df['session'] == s, 'C4env'] = np.abs(hilbert(fft_filter(df.loc[df['session'] == s, 'C4'], fs, band)))
    df.loc[df['session'] == s, 'SMRLeftEnv'] = np.abs(hilbert(fft_filter(np.dot(df.loc[df['session'] == s, channels], left), fs, band)))
    df.loc[df['session'] == s, 'SMRRightEnv'] = np.abs(hilbert(fft_filter(np.dot(df.loc[df['session'] == s, channels], right), fs, band)))
    #df.loc[df['session'] == s, 'C4env'] = np.abs(hilbert(x[:, channels.index('C4')]))
    #df.loc[df['session'] == s, 'SMRLeftEnv'] = np.abs(hilbert(np.dot(x, left)))
    #df.loc[df['session'] == s, 'SMRRightEnv'] = np.abs(hilbert(np.dot(x, right)))
    #plt.plot(df.loc[df['session']==s, 'C3env'])

df.to_pickle('{}day{}.pkl'.format(subj, day))

plt.plot(df['C3env'].rolling(fs*120).median())
plt.plot(df['C4env'].rolling(fs*120).median())
plt.plot(df['SMRLeftEnv'].rolling(fs*120).median())
plt.plot(df['SMRRightEnv'].rolling(fs*120).median())
plt.plot(df['C3env'].rolling(fs*120).median()+df['C4env'].rolling(fs*120).median())
plt.plot(df['SMRLeftEnv'].rolling(fs*120).median()+df['SMRRightEnv'].rolling(fs*120).median())
plt.legend(['C3', 'C4', 'SMRLeftEnv', 'SMRRightEnv', 'C4+C3', 'SMRLeftEnv+SMRRightEnv'])
plt.show()


