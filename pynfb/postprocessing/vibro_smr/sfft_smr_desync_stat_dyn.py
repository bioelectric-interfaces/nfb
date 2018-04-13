import pandas as pd
from pynfb.postprocessing.utils import load_data, runica2
from pynfb.signal_processing.helpers import get_outliers_mask
import numpy as np
from pynfb.inlets.montage import Montage
from mne.viz import plot_topomap
import pylab as plt
from scipy import signal

import seaborn as sns

from scipy.stats import ttest_ind, ttest_1samp, ttest_rel, wilcoxon, ranksums

wdir = '/home/nikolai/_Work/vibro_smr/'
data_dir = '/media/nikolai/D27ECFCB7ECFA697/Users/Nikolai/Desktop/vibro-decay/'
experiments = pd.read_csv(wdir + 'vibro-decay.csv')
experiments = experiments[experiments.protocol == 'belt']
print(experiments)


bands = [('alpha',  (8, 15)),  ('beta', (15, 20)), ('high-beta', (20, 28)), ('low-gamma', (30, 70))]
blocks_list = ['baseline-before', 'motor-before', 'rest-before', 'stimulation', 'rest-after', 'motor-after', 'baseline-after']
blocks_intervals = [(0, 2), (2, 4), (4, 6), (6, 7), (7, 9), (9, 11), (11, 13)]
blocks_dict = dict(zip(blocks_list, blocks_intervals))
print(blocks_dict)
#TODO add max desync. dreq.
metrics = ['mean desync.', 'median desync.', 'auc']

band_name = 'beta'
band = dict(bands)[band_name]


specs = pd.DataFrame(columns=['magnitude', 'condition', 'subj', 'time'])
print('******', len(experiments))
print(experiments['name'].unique())


for n_exp in list(range(0 , len(experiments)))[::-1]:
    print(n_exp)


    exp = experiments.iloc[n_exp]
    if n_exp == 0:
        continue
    if exp['name'] in ['va-ba']:
        print('\n'.join(['*********************************']*10))
        continue
    desc = '{}-{}-{}-{}'.format(exp['subject'], exp['protocol'], {0: 'exp', 1:'control'}[exp['control']], '-'.join(exp.dataset.split('_')[-2:]))
    print(exp, '\n*******************', desc, '\n*******************')
    df, fs, p_names, channels = load_data('{}{}/experiment_data.h5'.format(data_dir, exp.dataset))
    df.iloc[get_outliers_mask(df[channels], std=3, iter_numb=20)] = 0
    #channels = channels[:32]

    channels = channels[:32]

    right = np.load(wdir + desc + '-RIGHT.npy')[0]
    left = np.load(wdir + desc + '-LEFT.npy')[0]

    x = np.dot(df[channels], right)

    f, t, Sxx = signal.spectrogram(x, fs, scaling='spectrum', nperseg=fs*10, nfft=fs*10, noverlap=int(fs*10*0.90))
    Sxx = Sxx**0.5
    print(f.shape, t.shape, Sxx.shape)


    norm = np.mean(Sxx[(f >= band[0]) & (f <= band[1])][:, t <= 120])
    bandpow = np.mean(Sxx[ (f >= band[0]) & (f <= band[1])], 0)/norm
    specs = pd.concat([specs, pd.DataFrame({'magnitude': bandpow, 'condition': 'control' if exp['control'] else 'exp', 'subj': exp['name'],
                         'time': t})])



sns.tsplot(specs, 'time', 'subj', 'condition', 'magnitude')
plt.vlines(np.array([2, 4, 6, 7, 9, 11])*60, [-10]*6, [10]*6, alpha=0.5)
plt.plot([-100, 13*60], [1, 1], 'k--', alpha=0.5)
plt.title(band_name)
for j, (x, y) in enumerate(blocks_intervals):
    plt.text((x+y)/2*60, 0.2, blocks_list[j],  ha='center', va='bottom')

time = np.arange(0, 13*60, 5)


for j, t in enumerate(time[:-1]):
    real = specs.loc[(specs['condition'] == 'exp') & (specs['time'] >= time[j]) & (specs['time'] < time[j+1])].groupby('subj').mean()['magnitude']
    control = \
    specs.loc[(specs['condition'] == 'control') & (specs['time'] >= time[j]) & (specs['time'] < time[j+1])].groupby(
        'subj').mean()['magnitude']



    #p_v = ranksums(real, control).pvalue
    if p_v < 0.05:
        plt.fill_between([time[j], time[j+1]], -100, 100, color='g', alpha=0.3, linewidth=0.0 )

plt.xlim(0, 13*60)
plt.ylim(0, 2)

#sns.tsplot(specs, 'freq.', 'subj', 'condition', 'D', err_style='unit_traces')
plt.show()
