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

band_name = 'high-beta'

ch = ['SMR-R', 'SMR-L'][0]

comp = ['erd-mot', 'vib'][0]
if comp == 'erd-mot':
    block2_name = 'baseline-before'
    block1_name = 'motor-before'
    coef_name = '$ERD$ [%]'
elif comp == 'vib':
    block2_name = 'rest-before'
    block1_name = 'stimulation'
    coef_name = '$ERD(Vib)$ [%]'

#band = bands[band_name]
block2 = np.array(blocks_dict[block2_name])*60
block1 = np.array(blocks_dict[block1_name])*60
#print(band, block2, block1)


dfd = pd.DataFrame(columns=['100-percentile2', 'percentile1', 'condition', 'subj'])
aucs = pd.DataFrame(columns=['auc', 'condition', 'subj'])
des = pd.DataFrame(columns=[coef_name, 'condition', 'subj', 'band_name'])
specs = pd.DataFrame(columns=[coef_name, 'condition', 'subj', 'freq.'])
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
    #df = df[~get_outliers_mask(df[channels], std=3)]
    #df.iloc[get_outliers_mask(df[channels], std=3, iter_numb=20)] = 0
    #channels = channels[:32]

    channels = channels[:32]
    if fs == 1000:
        df = df.iloc[::2]
        fs = 500
    right = np.load(wdir + desc + '-RIGHT.npy')[0]
    left = np.load(wdir + desc + '-LEFT.npy')[0]

    x = np.dot(df[channels], right if ch == 'SMR-R' else left)

    #emg_channels = channels[32:]
    #m = {'fs': fs, 'eeg_channels': channels, 'emg_channels': ['APB_R', 'FDS_R', 'EDS_R', 'APB_L', 'FDS_L', 'EDS_L'],
    #     'blocks_names': p_names, 'blocks_course': df['block_number'].as_matrix(),
    #     'eeg': df[channels].as_matrix(), 'emg': df[emg_channels].as_matrix(), 'smr_right': np.dot(df[channels], right),
    #     'smr_left': np.dot(df[channels], left), 'ica_filter_left':left, 'ica_filter_right':right}


    #print(m)
    #from scipy.serializers import savemat
    #savemat('eeg_emg_smr.mat', m)





    f, t, Sxx = signal.spectrogram(x, fs, scaling='spectrum', nperseg=fs*10, nfft=fs*10, noverlap=10*int(fs*0.8))
    #print(Sxx)
    Sxx = Sxx**0.5
    print(f.shape, t.shape, Sxx.shape)


    #band = [20, 28]
    # desyncs = []
    # qs = np.linspace(0,100, 100)
    # for q in qs:
    #    rest = np.percentile(np.mean(Sxx[(f>=band[0]) & (f<=band[1])][:, (t>=block2[0]) & (t<=block2[1])], 0), q)
    #    motor = np.percentile(np.mean(Sxx[(f >= band[0]) & (f <= band[1])][:, (t >= block1[0]) & (t <= block1[1])], 0), q)
    #    desyncs.append((rest - motor)/(rest+motor)*2)
    # for q in qs:
    #     motor = np.percentile(np.mean(Sxx[(f >= band[0]) & (f <= band[1])][:, (t >= block1[0]) & (t <= block1[1])],0), q)
    #     rest = np.mean(Sxx[(f >= band[0]) & (f <= band[1])][:, (t >= block2[0]) & (t <= block2[1])], 0)
    #     desyncs.append(np.sum(rest > motor) / rest.shape[0] * 100)
    # auc = np.mean(np.array(desyncs))
    #
    # dfd = dfd.append(pd.DataFrame(
    #     {'100-percentile2': desyncs-np.linspace(100, 0, 100), 'percentile1': qs, 'condition': 'control' if exp['control'] else 'exp',
    #      'subj': exp['name']}))
    # aucs.loc[len(aucs)] = {'auc': auc, 'condition': 'control' if exp['control'] else 'exp', 'subj': exp['name']}

    rest = np.mean(Sxx[:, (t >= block2[0]) & (t <= block2[1])], 1)
    motor = np.mean(Sxx[:, (t >= block1[0]) & (t <= block1[1])], 1)
    desyn = (rest-motor) / rest
    specs = pd.concat([specs, pd.DataFrame({coef_name: desyn*100, 'condition': 'control' if exp['control'] else 'exp', 'subj': exp['name'],
                         'freq.': f})])

    for band_name, band in bands:

        rest = np.median(np.mean(Sxx[(f>=band[0]) & (f<=band[1])][:, (t>=block2[0]) & (t<=block2[1])], 0))
        motor = np.median(np.mean(Sxx[(f >= band[0]) & (f <= band[1])][:, (t >= block1[0]) & (t <= block1[1])], 0))
        desyn = (rest-motor)  / rest
        des.loc[len(des)] = {coef_name: desyn*100, 'condition': 'control' if exp['control'] else 'exp', 'subj': exp['name'], 'band_name':band_name}
    # print('auc', np.mean(desyncs))

    #sns.kdeplot(np.mean(Sxx[(f>=band[0]) & (f<=band[1])][:, (t>=block2[0]) & (t<=block2[1])], 0), )
    #sns.kdeplot(np.mean(Sxx[(f >= band[0]) & (f <= band[1])][:, (t >= block1[0]) & (t <= block1[1])], 0))
    #plt.show()

    print(dfd)
    # plt.plot(qs, desyncs)
    # plt.ylim(0, 100)
    # plt.xlim(0, 100)
    # plt.xlabel('Percentile({})'.format(block1_name))
    # plt.ylabel('1 - Percentile({})'.format(block2_name))
    # plt.title(exp['name'] + '-' + desc)
    # plt.legend(['AUC = {}'.format(np.mean(desyncs))])
    # plt.plot([0, 100], [100, 0], 'k--')
    # plt.show()

# sns.tsplot(dfd, 'percentile1', 'subj', condition='condition', value='100-percentile2', estimator=np.median)
# sns.tsplot(dfd, 'percentile1', 'subj', condition='condition', value='100-percentile2', err_style="unit_traces", estimator=np.median)
# plt.title('Rest before VS. Motor before')
# plt.ylim(-1, 1.5)
# plt.plot([0, 100], [100, 0], 'k--')
# plt.show()

# sns.boxplot(x='condition', y='auc', data=aucs)
# sns.swarmplot(x='condition', y='auc', data=aucs, color='k', alpha=0.5)
# print(aucs.loc[aucs['condition']=='exp'])
# print(aucs.loc[aucs['condition']=='control'])
# print(aucs.loc[aucs['condition']=='exp', 'auc'].as_matrix() - aucs.loc[aucs['condition']=='control', 'auc'].as_matrix())
# print(ttest_ind(aucs.loc[aucs['condition']=='exp', 'auc'], aucs.loc[aucs['condition']=='control', 'auc']))
# print(ttest_rel(aucs.loc[aucs['condition']=='exp', 'auc'], aucs.loc[aucs['condition']=='control', 'auc']))
# print(wilcoxon(aucs.loc[aucs['condition']=='exp', 'auc'], aucs.loc[aucs['condition']=='control', 'auc']))
# print(ranksums(aucs.loc[aucs['condition']=='exp', 'auc'], aucs.loc[aucs['condition']=='control', 'auc']))
# print(ttest_1samp(aucs.loc[aucs['condition']=='exp', 'auc'], 50))
# print(ttest_1samp(aucs.loc[aucs['condition']=='control', 'auc'], 50))
# plt.show()


sns.boxplot(x='band_name', y=coef_name, hue='condition',  data=des)
plt.ylim(-30, 80)
plt.title(ch)

from rpy2.robjects import r as r
from rpy2 import robjects



for j, (band_name, band) in enumerate(bands):
    des_band = des.loc[des['band_name'] == band_name]
    #p_v = wilcoxon(des_band.loc[des['condition']=='exp', coef_name], des_band.loc[des['condition']=='control', coef_name]).pvalue

    control = des_band.loc[des['condition']=='control', coef_name].tolist()
    real = des_band.loc[des['condition'] == 'exp', coef_name].tolist()
    p_v = r['wilcox.test'](robjects.FloatVector(real), robjects.FloatVector(control), paired=True)[2][0]
    print('control', ttest_1samp(control, 0).pvalue / 2)
    print('exp', ttest_1samp(real, 0).pvalue / 2)

    plt.text(j, 80, "{} {:.3f}".format('ns' if p_v > 0.05 else ('**' if p_v<0.01 else '*'), p_v), ha='center', va='bottom')
#sns.swarmplot(x='band_name', y=coef_name, hue='condition', data=des, color='k', alpha=0.5)
from scipy.stats import ttest_ind
#print(ttest_ind(des.loc[des['condition']=='exp', coef_name], des.loc[des['condition']=='control', coef_name]))
#print(ttest_rel(des.loc[des['condition']=='exp', coef_name], des.loc[des['condition']=='control', coef_name]))
#print(wilcoxon(des.loc[des['condition']=='exp', coef_name], des.loc[des['condition']=='control', coef_name]))
#plt.title('Ranksums p-value:')
print(wilcoxon(des.loc[des['condition']=='exp', coef_name], des.loc[des['condition']=='control', coef_name]))
plt.show()