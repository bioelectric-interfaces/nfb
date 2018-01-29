from pynfb.postprocessing.mu_or_not.meta import *
subj = 'p4'
data = pd.DataFrame()
subjects = ['p6', 'p8', 'p13', 'p10', 'VV']
#subjects = ['George', 'Maria', 'p14', 'p15', 'p18', 'p20', 'p21']

import seaborn as sns


df = pd.DataFrame(columns=['val', 'type', 'subj', 'session'])
for subj in subjects:
    subj_data = pd.DataFrame(columns=['c3env', 'c4env', 'session'])
    for day in [1, 2]:
        eeg = []
        sessions = []
        for k in range(15):
            print(r'{0}\treatment\{1}\day{3}\proba{2}.mat'.format(data_dir, subj, k + 1, day))
            mat = loadmat(r'{0}\treatment\{1}\day{3}\proba{2}.mat'.format(data_dir, subj, k + 1, day))['X1Topo'].T
            clear = mat[~get_outliers_mask(mat, iter_numb=20, std=3)]
            eeg.append(clear)
            sessions.append([k + 1] * len(eeg[-1]))
        dd = pd.DataFrame(columns=channels, data=np.concatenate(eeg))
        dd['session'] = np.concatenate(sessions)
        dd['session'] += 15 * (day - 1)
        dd['c3env'] = dd['C3'] * 0
        dd['c4env'] = dd['C4'] * 0
        dd['rightenv'] = dd['C3'] * 0
        dd['leftenv'] = dd['C4'] * 0

        filters = np.load(r'treatment\{0}d{1}_filters.npy'.format(subj, day))
        topos = np.load(r'treatment\{0}d{1}_topographies.npy'.format(subj, day))
        ind = np.load(r'treatment\{0}d{1}_smr_ind.npy'.format(subj, day))
        ch = [c for c in channels if c != 'Pz']
        left = filters[:, ind[np.argmax(np.abs(topos[ch.index('C4'), ind]))]]
        right = filters[:, ind[np.argmax(np.abs(topos[ch.index('C3'), ind]))]]
        #channels.remove('Pz')
        for s in dd['session'].unique():
            dd.loc[dd['session'] == s, 'c3env'] = np.abs(hilbert(fft_filter(dd.loc[dd['session'] == s, 'C3'], fs, band)))
            dd.loc[dd['session'] == s, 'c4env'] = np.abs(hilbert(fft_filter(dd.loc[dd['session'] == s, 'C4'], fs, band)))
            dd.loc[dd['session'] == s, 'leftenv'] = np.abs(hilbert(fft_filter(np.dot(dd.loc[dd['session'] == s, ch], left), fs, band)))
            dd.loc[dd['session'] == s, 'rightenv'] = np.abs(hilbert(fft_filter(np.dot(dd.loc[dd['session'] == s, ch], right), fs, band)))
        dd = dd[['c3env', 'c4env', 'leftenv', 'rightenv', 'session']]
        #dd['session'] = (dd['session'] - 1) // 3 + 1
        dd = dd.groupby('session').mean()
        dd['session'] = dd.index
        print(dd)
        subj_data = subj_data.append(dd)
    #subj_data.loc[:, ['c3env', 'c4env']] /= subj_data.loc[subj_data['session'] == 1, ['c3env', 'c4env']].mean()
    for type in ['c3env', 'c4env']:
        df = df.append(pd.DataFrame({'session': subj_data['session'], 'val': subj_data[type], 'type':type, 'subj': subj}))
    print(df)
        #for val in ['SMRRightEnv', 'SMRLeftEnv', 'C3env', 'C4env']:
         #   df = df.append(pd.DataFrame({'val': dd[val], 'subj': subj, 'day': day, 'session': dd['session'] + (day-1)*5, 'type':val}))

df.to_pickle('treatment_smr.pkl')
sns.boxplot('session', 'val', 'type', df)
sns.swarmplot('session', 'val', 'type', df)
plt.show()
