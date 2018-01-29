from pynfb.postprocessing.mu_or_not.meta import *
subj = 'p4'
data = pd.DataFrame()
subjects = ['p4', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p22', 'p23', 'p24', 'p25', 'p6', 'VV', 'VG', 'IO', 'KM']
subjects = ['George', 'Maria', 'p14', 'p15', 'p18', 'p20', 'p21']

import seaborn as sns


from scipy import fftpack
from numpy.fft import fftfreq
import numpy as np

def hilbert_env(x, fs, band, N=None, axis=-1):
    x = np.asarray(x)
    Xf = fftpack.fft(x, N, axis=axis)
    w = fftfreq(x.shape[0], d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = fftpack.ifft(Xf, axis=axis)
    return np.abs(2*x)

df = pd.DataFrame(columns=['val', 'type', 'subj', 'session'])
for subj in subjects:
    subj_data = pd.DataFrame(columns=['c3env', 'c4env', 'session'])
    for day in [1, 2]:
        eeg = []
        sessions = []
        for k in range(15):
            print(r'{0}\treatment\{1}\day{3}\proba{2}.mat'.format(data_dir, subj, k + 1, day))
            mat = loadmat(r'{0}\control\{1}\day{3}\proba{2}.mat'.format(data_dir, subj, k + 1, day))['X1Topo'].T
            clear = mat[~get_outliers_mask(mat, iter_numb=20, std=3)]
            eeg.append(clear)
            sessions.append([k + 1] * len(eeg[-1]))
        dd = pd.DataFrame(columns=channels, data=np.concatenate(eeg))
        dd['session'] = np.concatenate(sessions)
        dd['session'] += 15 * (day - 1)
        dd['c3env'] = dd['C3'] * 0
        dd['c4env'] = dd['C4'] * 0
        for s in dd['session'].unique():
            dd.loc[dd['session'] == s, 'c3env'] = hilbert_env(dd.loc[dd['session'] == s, 'C3'], fs, band)
            dd.loc[dd['session'] == s, 'c4env'] = hilbert_env(dd.loc[dd['session'] == s, 'C4'], fs, band)
        dd = dd[['c3env', 'c4env', 'session']]
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

df.to_pickle('control_c3c4.pkl')
sns.boxplot('session', 'val', 'type', df)
sns.swarmplot('session', 'val', 'type', df)
plt.show()
