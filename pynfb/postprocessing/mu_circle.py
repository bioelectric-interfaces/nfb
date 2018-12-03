from pynfb.serializers.xml_ import get_lsl_info_from_xml
import numpy as np
import pandas as pd
import seaborn as sns


pilot_dir = 'C:\\Users\\Nikolai\\Downloads\\pilot'

experiments = ['pilot_2mok_nikolay_10-18_15-25-45',
               'pilot_2mok_Plackhin_10-20_11-54-00',
               'pilot_2mok_Tatiana_10-18_15-56-49',
               'pilot_2mok_Polyakova_10-24_15-16-52']

experiments_main = ['pilot_Nikolay_2_10-18_14-57-23',
                    'pilot_Plackhin_1_10-20_12-03-01',
                    'pilot_Tatiana_2_10-18_16-00-44',
                    'pilot_Polyakova_1_10-24_15-21-18']


import h5py
results = {}
labels = None
fs = None

channels = ['C3', 'C4']
for experiment, experiment_main in zip(experiments, experiments_main):
    with h5py.File('{}\\{}\\{}'.format(pilot_dir, experiment_main, 'experiment_data.h5')) as f:
        rejections = [f['protocol1/signals_stats/left/rejections/rejection{}'.format(j+1)][:] for j in range(2)]
        rejection = np.dot(rejections[0], rejections[1])

    with h5py.File('{}\\{}\\{}'.format(pilot_dir, experiment, 'experiment_data.h5')) as f:

        raw_data = [np.dot(f['protocol{}/raw_data'.format(j+1)][:], rejection) for j in range(3)]
        labels_, fs_ = get_lsl_info_from_xml(f['stream_info.xml'][0])
        if labels:
            assert labels == labels_, 'Labels should be the same'
        if fs:
            assert fs == fs_, 'FS should be the same'
        labels = labels_
        fs = fs_
    results[experiment] = [np.array([raw[:, labels.index(channel)] for channel in channels]) for raw in raw_data]

import pylab as plt
plt.plot(results[experiments[3]][1][0])
plt.show()

from scipy.stats import ttest_ind, ttest_1samp

f = plt.figure()
channel = 'C3'
all_pows = []
from scipy.fftpack import rfft, irfft, fftfreq
for exp_j, experiment in enumerate(experiments):
    print(experiment)
    n_windows = 30
    pows = np.zeros((3, n_windows))
    for protocol_j in range(3):
        data = results[experiment][protocol_j][channels.index(channel), :]
        n_taps = 2000
        for ind, j in enumerate(range(0, len(data) - n_taps, len(data)//n_windows)):
            f_signal = rfft(data[j:j+n_taps])
            w = fftfreq(n_taps, d=1. / fs*2)
            cut_f_signal = f_signal.copy()
            bandpass = (9, 14)
            cut_f_signal[(w < bandpass[0]) | (w > bandpass[1])] = 0
            pows[protocol_j, ind] += np.abs(cut_f_signal**2).mean()
            print(sum((w > bandpass[0]) & (w < bandpass[1])))
    n_pows = pows / pows.mean(1)[0]
    plt.errorbar(np.arange(3)+ 0.05 *(-1 + exp_j), n_pows.mean(1), n_pows.std(1))
    plt.xticks([0, 1, 2], ['background', 'random border', 'sin border'])
    plt.xlim(-0.5, 2.5)
    plt.ylabel('power / background power')
    plt.title('Relative mean power in {} with rejections: ICA_artifacts and CSP_alpha'.format(channel)) # with rejections: ICA_artifacts and CSP_alpha
    print(n_pows.mean(1))
    print(n_pows.std(1))
    all_pows.append(n_pows)

all_pows = np.array(all_pows)

print(all_pows[:,0])
print(ttest_1samp(all_pows[:,0].flatten(), 1))
print(ttest_1samp(all_pows[:,1].flatten(), 1))
print(ttest_1samp(all_pows[:,2].flatten(), 1))
plt.legend(experiments)
plt.savefig(channel+'.png', dpi=200)


plt.show()

plt.figure()
plt.hist(all_pows[:,0].flatten(), bins=30)

plt.show()

plt.figure()
plt.hist(all_pows[:,1].flatten(), bins=30)
plt.show()

plt.figure()
plt.hist(all_pows[:,2].flatten(), bins=30)
plt.show()