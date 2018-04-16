import pandas as pd
from pynfb.postprocessing.utils import load_data, runica2
from pynfb.signal_processing.helpers import get_outliers_mask
import numpy as np
from pynfb.inlets.montage import Montage
from mne.viz import plot_topomap
import pylab as plt
from scipy import signal

import seaborn as sns


wdir = '/home/nikolai/_Work/vibro_smr/'
data_dir = '/media/nikolai/D27ECFCB7ECFA697/Users/Nikolai/Desktop/vibro-decay/'
experiments = pd.read_csv(wdir + 'vibro-decay.csv')
experiments = experiments[experiments.protocol == 'belt']
print(experiments)


bands = {'alpha': (8, 15), 'low-beta': (12, 16), 'beta': (16, 20), 'high-beta': (20, 28), 'low-gamma': (29, 70)}
blocks_list = ['baseline-before', 'motor-before', 'rest-before', 'stimulation', 'rest-after', 'motor-after', 'baseline-after']
blocks_intervals = [(0, 2), (2, 4), (4, 6), (6, 7), (7, 9), (9, 11), (11, 13)]
blocks_dict = dict(zip(blocks_list, blocks_intervals))
print(blocks_dict)
#TODO add max desync. dreq.
metrics = ['mean desync.', 'median desync.', 'auc']

band_name = 'alpha'
block2_name = 'rest-before'
block1_name = 'motor-before'

band = bands[band_name]
block2 = np.array(blocks_dict[block2_name])*60
block1 = np.array(blocks_dict[block1_name])*60
print(band, block2, block1)


dfd = pd.DataFrame(columns=['100-percentile2', 'percentile1', 'condition', 'subj'])
des = pd.DataFrame(columns=['desync.', 'condition', 'subj'])

for n_exp in [5]:
    print(n_exp)


    exp = experiments.iloc[n_exp]

    desc = '{}-{}-{}-{}'.format(exp['subject'], exp['protocol'], {0: 'exp', 1:'control'}[exp['control']], '-'.join(exp.dataset.split('_')[-2:]))
    print(exp, '\n*******************', desc, '\n*******************')
    df, fs, p_names, channels = load_data('{}{}/experiment_data.h5'.format(data_dir, exp.dataset))
    # df = df[~get_outliers_mask(df[channels], std=3)]

    right = np.load(wdir + desc + '-RIGHT.npy')[0]
    x = np.dot(df[channels], right)
    t = np.arange(len(x)) / fs

    snrs = np.linspace(0.75, 5, 100)
    aucs = np.zeros_like(snrs)
    erds = np.zeros_like(snrs)
    erdds = np.zeros_like(snrs)
    noise = np.random.normal(x.mean(), x.std(), len(x))
    for j, snr in enumerate(snrs):
        #x += noise*snr
        #block_slice =
        #x0 = x[(t >= block1[0]) & (t <= block1[1]) & (t >= block1[0]) & (t <= block1[1])]
        #x0[np.random.randint(0, len(x0), j*1)] *= 10


        #plt.plot(x)
        #plt.show()
        x1 = x[(t >= block1[0]) & (t <= block1[1])]
        x2 = x[(t >= block2[0]) & (t <= block2[1])]
        x_all = np.concatenate([x1, x2])
        #for k in range(j):
        #    rand_start = np.random.randint(0, len(x_all)-500)
        #    x_all[rand_start:rand_start+500] += np.random.normal(x.mean(), x.std(), 500)*5
        np.random.seed(42)
        x_all += np.random.normal(x_all.mean(), x_all.std(), len(x_all)) * snr
        #plt.plot(x_all)
        #plt.show()

        x1 = x_all[:len(x1)]
        x2 = x_all[len(x1):]



        f1, t1, Sxx1 = signal.spectrogram(x1, fs, scaling='spectrum', nfft=fs * 2)
        f2, t2, Sxx2 = signal.spectrogram(x2, fs, scaling='spectrum', nfft=fs * 2)

        # roc auc desync.
        rest_power = np.mean(Sxx1[(f1 >= band[0]) & (f1 <= band[1])],0)
        motor_power = np.mean(Sxx2[(f2 >= band[0]) & (f2 <= band[1])], 0)
        #plt.plot(motor_power)
        #plt.plot(rest_power)
        #plt.show()


        roc = []
        qs = np.linspace(0,100, 100)
        for q in qs:
            th = np.percentile(rest_power, 100-q)
            roc.append(np.sum(motor_power > th) / motor_power.shape[0] * 100)


        # ERD-d

        from scipy import stats
        bins, h = np.linspace(0, max(rest_power.max(), motor_power.max()), 100, retstep=True)
        motor_h= stats.gaussian_kde(motor_power)(bins)
        rest_h = stats.gaussian_kde(rest_power)(bins)
        motor_h = motor_h / motor_h.sum()
        rest_h = rest_h / rest_h.sum()

        print(motor_h.sum())
        erdd = sum(rest_h[motor_h>rest_h]) + sum(motor_h[motor_h<rest_h])
        erdds[j] = 1 - erdd




        #plt.plot(qs, roc)
        #plt.show()
        auc = np.mean(np.array(roc))
        aucs[j] = auc


        # Pfurtscheller ERD
        rest = np.mean(np.mean(Sxx2[(f2>=band[0]) & (f2<=band[1])], 0))
        motor = np.mean(np.mean(Sxx1[(f1 >= band[0]) & (f1 <= band[1])], 0))
        erd = (rest - motor)/rest
        erds[j] = erd

    print(dfd)
    f, ax = plt.subplots(3)
    ax[0].plot(1/snrs, aucs)
    ax[0].set_ylabel('AUC-ERD')
    ax[1].plot(1/snrs, erds)
    ax[1].set_ylabel('ERD')
    ax[2].plot(1 / snrs, erdds)
    ax[2].set_ylabel('ERD-d')


    plt.show()
    plt.plot(1/snrs, (aucs-50)/(aucs[0]-50))
    #ax[3].plot(snrs, (merds)/merds[0])
    plt.plot(1/snrs, (erds) / erds[0])
    plt.plot(1/snrs, (erdds) / erdds[0])
    plt.ylabel('Normalized')
    plt.legend(['AUC-ERD', 'ERD', 'ERD-d'])
    plt.xlabel('SNR')
    plt.show()

    # plt.plot(qs, roc)
    # plt.ylim(0, 100)
    # plt.xlim(0, 100)
    # plt.xlabel('Percentile({})'.format(block1_name))
    # plt.ylabel('1 - Percentile({})'.format(block2_name))
    # plt.title(exp['name'] + '-' + desc)
    # plt.legend(['AUC = {}'.format(np.mean(roc))])
    # plt.plot([0, 100], [0, 100], 'k--')
    # plt.show()

