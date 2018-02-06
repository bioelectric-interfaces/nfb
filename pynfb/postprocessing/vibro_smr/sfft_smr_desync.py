import pandas as pd
from pynfb.postprocessing.utils import load_data, runica2
from pynfb.signal_processing.helpers import get_outliers_mask
import numpy as np
from pynfb.inlets.montage import Montage
from mne.viz import plot_topomap
import pylab as plt
from scipy import signal

wdir = '/home/nikolai/_Work/vibro_smr/'
data_dir = '/media/nikolai/D27ECFCB7ECFA697/Users/Nikolai/Desktop/vibro-decay/'
experiments = pd.read_csv(wdir + 'vibro-decay.csv')
experiments = experiments[experiments.protocol == 'belt']
print(experiments)


dfd = pd.DataFrame(columns=['100-percentile2', 'percentile1', 'condition', 'subj'])
aucs = pd.DataFrame(columns=['auc', 'condition', 'subj'])
des = pd.DataFrame(columns=['desync.', 'condition', 'subj'])

for n_exp in range(0 , len(experiments)):
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
    # df = df[~get_outliers_mask(df[channels], std=3)]


    right = np.load(wdir + desc + '-RIGHT.npy')[0]
    left = np.load(wdir + desc + '-LEFT.npy')[0]

    x = np.dot(df[channels], right)


    f, t, Sxx = signal.spectrogram(x, fs, scaling='spectrum', nfft=fs*2)
    #print(Sxx)
    Sxx = Sxx**0.5
    print(f.shape, t.shape, Sxx.shape)


    band = [20, 28]
    rest_t = [7*60, 9*60]
    motor_t = [4*60, 6*60]
    desyncs = []
    qs = np.linspace(0,100, 100)
    #for q in qs:
    #    rest = np.percentile(np.mean(Sxx[(f>=band[0]) & (f<=band[1])][:, (t>=rest_t[0]) & (t<=rest_t[1])], 0), q)
    #    motor = np.percentile(np.mean(Sxx[(f >= band[0]) & (f <= band[1])][:, (t >= motor_t[0]) & (t <= motor_t[1])], 0), q)
    #    desyncs.append((rest - motor)/(rest+motor)*2)
    for q in qs:
        motor = np.percentile(np.mean(Sxx[(f >= band[0]) & (f <= band[1])][:, (t >= motor_t[0]) & (t <= motor_t[1])],0), q)
        rest = np.mean(Sxx[(f >= band[0]) & (f <= band[1])][:, (t >= rest_t[0]) & (t <= rest_t[1])], 0)
        desyncs.append(np.sum(rest > motor) / rest.shape[0] * 100)
    auc = np.mean(np.array(desyncs))

    dfd = dfd.append(pd.DataFrame(
        {'100-percentile2': desyncs, 'percentile1': qs, 'condition': 'control' if exp['control'] else 'exp',
         'subj': exp['name']}))
    aucs.loc[len(aucs)] = {'auc': auc, 'condition': 'control' if exp['control'] else 'exp', 'subj': exp['name']}
    rest = np.median(np.mean(Sxx[(f>=band[0]) & (f<=band[1])][:, (t>=rest_t[0]) & (t<=rest_t[1])], 0))
    motor = np.median(np.mean(Sxx[(f >= band[0]) & (f <= band[1])][:, (t >= motor_t[0]) & (t <= motor_t[1])], 0))
    desyn = (rest - motor)*2/(rest + motor)
    des.loc[len(des)] = {'desync.': desyn, 'condition': 'control' if exp['control'] else 'exp', 'subj': exp['name']}
    #print('auc', np.mean(desyncs))

    #print(dfd)
    #plt.plot(qs, desyncs)
    #plt.ylim(0, 100)
    #plt.xlim(0, 100)
    #plt.xlabel('Percentile(Motor)')
    #plt.ylabel('1 - Percentile(Rest)')
    #plt.title(exp['name'] + '-' + desc)
    #plt.legend(['AUC = {}'.format(np.mean(desyncs))])
    #plt.plot([0, 100], [100, 0], 'k--')
    #plt.show()

import seaborn as sns
sns.tsplot(dfd, 'percentile1', 'subj', condition='condition', value='100-percentile2', estimator=np.median)
sns.tsplot(dfd, 'percentile1', 'subj', condition='condition', value='100-percentile2', err_style="unit_traces", estimator=np.median)
plt.title('Rest before VS. Motor before')
plt.ylim(-1, 1.5)
plt.plot([0, 100], [100, 0], 'k--')
plt.show()

sns.boxplot(x='condition', y='auc', data=aucs)
sns.swarmplot(x='condition', y='auc', data=aucs, color='k', alpha=0.5)
from scipy.stats import ttest_ind, ttest_1samp
print(ttest_ind(aucs.loc[aucs['condition']=='exp', 'auc'], aucs.loc[aucs['condition']=='control', 'auc']))
print(ttest_1samp(aucs.loc[aucs['condition']=='exp', 'auc'], 50))
print(ttest_1samp(aucs.loc[aucs['condition']=='control', 'auc'], 50))
plt.show()

sns.boxplot(x='condition', y='desync.', data=des)
sns.swarmplot(x='condition', y='desync.', data=des, color='k', alpha=0.5)
from scipy.stats import ttest_ind
print(ttest_ind(des.loc[des['condition']=='exp', 'desync.'], des.loc[des['condition']=='control', 'desync.']))
plt.show()