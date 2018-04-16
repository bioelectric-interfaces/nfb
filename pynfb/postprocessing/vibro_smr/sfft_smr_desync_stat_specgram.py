import pandas as pd
from pynfb.postprocessing.utils import load_data, runica2
from pynfb.signal_processing.helpers import get_outliers_mask
import numpy as np
from pynfb.inlets.montage import Montage
from mne.viz import plot_topomap
import pylab as plt
from scipy import signal

import seaborn as sns
cm = sns.color_palette()

from scipy.stats import ttest_ind, ttest_1samp, ttest_rel, wilcoxon, ranksums

wdir = '/home/nikolai/_Work/vibro_smr/'
data_dir = '/media/nikolai/D27ECFCB7ECFA697/Users/Nikolai/Desktop/vibro-decay/'
experiments = pd.read_csv(wdir + 'vibro-decay.csv')
experiments = experiments[experiments.protocol == 'belt']
print(experiments)


bands = [('alpha',  (8, 15)),  ('beta', (15, 20)), ('high-beta', (20, 28)), ('low-gamma', (28, 70))]
blocks_list = ['REST', 'MOTOR', 'REST', 'VIB', 'REST', 'MOTOR', 'REST']
blocks_intervals = [(0, 2), (2, 4), (4, 6), (6, 7), (7, 9), (9, 11), (11, 13)]
blocks_dict = dict(zip(blocks_list, blocks_intervals))
print(blocks_dict)
#TODO add max desync. dreq.
metrics = ['mean desync.', 'median desync.', 'auc']

band_name = 'alpha'
band = dict(bands)[band_name]


print('******', len(experiments))
print(experiments['name'].unique())

all_specs = [None, None]
for control in [0, 1]:
    specs = []
    for n_exp in list(range(0 , len(experiments)))[:]:
        print(n_exp)


        exp = experiments.iloc[n_exp]
        if n_exp == 0:
            continue
        if exp['name'] in ['va-ba']:
            print('\n'.join(['*********************************']*10))
            continue
        if exp['control'] != control:
            continue
        desc = '{}-{}-{}-{}'.format(exp['subject'], exp['protocol'], {0: 'exp', 1:'control'}[exp['control']], '-'.join(exp.dataset.split('_')[-2:]))
        print(exp, '\n*******************', desc, '\n*******************')
        df, fs, p_names, channels = load_data('{}{}/experiment_data.h5'.format(data_dir, exp.dataset))
        #df.iloc[get_outliers_mask(df[channels], std=3, iter_numb=3)] = 0
        #channels = channels[:32]

        channels = channels[:32]
        if fs == 1000:
            df = df.iloc[::2]
            fs = 500

        df = df.iloc[:13 * fs * 60]

        right = np.load(wdir + desc + '-RIGHT.npy')[0]
        left = np.load(wdir + desc + '-LEFT.npy')[0]

        x = np.dot(df[channels], left)

        f, t, Sxx = signal.spectrogram(x, fs, scaling='spectrum', nperseg=fs*10, nfft=fs*10, noverlap=10*int(fs*0.8))
        Sxx = Sxx**0.5
        print(f.shape, t.shape, Sxx.shape)


        norm = np.median(Sxx[:, t <= 120], 1)
        bandpow = (Sxx/norm[:, None] - 1)*100

        # mot = np.median(Sxx[:, (t <= 240) & (t > 120)], 1)
        specs.append(bandpow)

        # fig = plt.figure()
        # ax = plt.pcolormesh(t / 60, f, (bandpow ** 0.5), cmap='RdBu_r', vmin=0.5, vmax=1.5)
        #
        # plt.xlabel('Time [s]')
        # plt.ylabel('Frequency [Hz]')
        #
        # for j, (x, y) in enumerate(blocks_intervals):
        #     plt.text((x + y) / 2, 35, blocks_list[j], ha='center', va='bottom')
        #     plt.axvline(y, color='k', linestyle='--', alpha=0.6)
        #
        # for (name, band) in bands:
        #     plt.text(1, np.mean(band), name, ha='center', va='center')
        #     plt.axhline(band[0], color='k', linestyle='--', alpha=0.6)
        #
        # plt.title(desc)
        # plt.ylim(0, 40)
        # plt.xlim(0, 13)
        #
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.10, 0.05, 0.8])
        # cb = fig.colorbar(ax, cax=cbar_ax)
        # cbar_ax.set_ylabel('Synchronization to the 1st REST')
        # fig.savefig('sync'+desc, dpi=200)
        # plt.show()
        # plt.close()



    fig = plt.figure(figsize=(8, 5))
    ax = plt.pcolormesh(t/60, f, np.median(specs, 0), cmap='RdBu_r', vmin=-80, vmax=80)

    plt.xlabel('Time $t$ [s]')
    plt.ylabel('Frequency $f$ [Hz]')

    for j, (x, y) in enumerate(blocks_intervals):

        plt.text((x+y)/2, 35, blocks_list[j],  ha='center', va='bottom', weight='bold')
        plt.axvline(y, color='k', linestyle='--', alpha=0.6)

    for (name, band) in bands:
        plt.text(1 , np.mean(band), name,  ha='center', va='center', weight='bold')
        plt.axhline(band[0], color='k', linestyle='--', alpha=0.6)

    plt.title('Experimental' if not control else 'Control')
    plt.ylim(0, 40)
    plt.xlim(0, 13)


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.10, 0.05, 0.8])
    cb = fig.colorbar(ax, cax=cbar_ax)
    cbar_ax.set_ylabel('$S(t, f)$ [%]')

    plt.show()

    all_specs[control] = specs



for band_name, band in bands:


    fig = plt.figure()
    plt.title(band_name)
    for c in [0, 1]:
        band_spec = np.mean(np.array(all_specs[c])[:, (f >= band[0]) & (f <= band[1])], 1)
        plt.plot(t/60, np.median(band_spec, 0), c=cm[c])
        p75 = np.percentile(band_spec, 75, 0)
        p25 = np.percentile(band_spec, 25,  0)
        plt.fill_between(t/60, p25, p75, alpha=0.3, color=cm[c])


    real = np.mean(np.array(all_specs[0])[:, (f >= band[0]) & (f <= band[1])], 1)
    control = np.mean(np.array(all_specs[1])[:, (f >= band[0]) & (f <= band[1])], 1)
    dt = (t[1] - t[0])/2
    p_vals = []
    for j, time in enumerate(t):
        p_v = ranksums(real[:, j], control[:, j]).pvalue
        if p_v < 0.05:
        #    #plt.axvline(time/60, color=cm[2], alpha=0.5)
            plt.fill_between(np.array([time-dt, time+dt])/60, -100, 100, color='g', alpha=0.2, linewidth=0.0 )


    for j, (x, y) in enumerate(blocks_intervals):
        plt.text((x+y)/2, 75, blocks_list[j],  ha='center', va='bottom')
        plt.axvline(y, color='k', linestyle='--', alpha=0.6)
    plt.xlim(0, 13)
    plt.ylim(-100, 100)
    plt.xlabel('Time $t$ [s]')
    plt.ylabel('$S_{band}(t)$ [%]')
    plt.show()