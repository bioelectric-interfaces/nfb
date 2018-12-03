from pynfb.serializers.xml_ import get_lsl_info_from_xml
import numpy as np
import pylab as plt

from scipy.signal import *

import pandas as pd
import seaborn as sns

def dc_blocker(x, r=0.99):
    # DC Blocker https://ccrma.stanford.edu/~jos/fp/DC_Blocker.html
    y = np.zeros_like(x)
    for n in range(1, x.shape[0]):
        y[n] = x[n] - x[n-1] + r * y[n-1]
    return y


pilot_dir = 'C:\\Users\\Nikolai\\Downloads\\pilot'


experiments1 = ['pilot_Nikolay_1_10-17_13-57-56',
                'pilot_Plackhin_1_10-20_12-03-01',
                'pilot_Tatiana_1_10-17_15-04-39',
                'pilot_Polyakova_1_10-24_15-21-18']

experiments2 = ['pilot_Nikolay_2_10-18_14-57-23',
                'pilot_Tatiana_2_10-18_16-00-44']

experiments = experiments1


import h5py
results = {}
labels = None
fs = None

channel = 'C3'
n_samples = 7500

for experiment in experiments:
    with h5py.File('{}\\{}\\{}'.format(pilot_dir, experiment, 'experiment_data.h5')) as f:
        rejections = [f['protocol1/signals_stats/left/rejections/rejection{}'.format(j+1)][:] for j in range(2)]
        rejection = rejections[0]#np.dot(rejections[0], rejections[1])

    with h5py.File('{}\\{}\\{}'.format(pilot_dir, experiment, 'experiment_data.h5')) as f:

        raw_data = [np.dot(f['protocol{}/raw_data'.format(j)][:], rejection) for j in [3, 2, 14]]

        raw_data = [raw_data[0][:7500], raw_data[1][:7500], raw_data[2][:7500], raw_data[1][7500:15000], raw_data[2][7500:15000]]
        print(raw_data)
        print([f['protocol{}'.format(j)].attrs['name'] for j in [3, 2, 14]])

        labels_, fs_ = get_lsl_info_from_xml(f['stream_info.xml'][0])
        labels_ = [label for label in labels_ if label not in ['A1', 'A2', 'AUX']]
        n_samples_ = min([len(raw_data[k]) for k in range(len(raw_data))])
        if labels:
            assert labels == labels_, 'Labels should be the same'
        if fs:
            assert fs == fs_, 'FS should be the same'
        labels = labels_
        fs = fs_


        results[experiment] = np.array([raw[:, labels.index(channel)] for raw in raw_data]).T
        print(results[experiment][0].shape)

print('asfaef', results)
b, a = butter(3, [9 / fs * 2, 14 / fs * 2], 'band')
f = plt.figure()

axs = [f.add_subplot(5, 1, k+1) for k in range(5)]
for experiment in experiments:
    x = filtfilt(b, a, results[experiment], axis=0)
    x = (x - x[:, 0].mean()) / x[:, 0].std()
    results[experiment] = x
    for k in range(5):
        axs[k].plot(x[:, k])
        axs[k].set_ylim(-4, 4)
plt.show()

powers = {}
n_windows = 8
n_taps = n_samples//n_windows
print('n_taps', n_taps)
for experiment in experiments:
    x = results[experiment]
    pow_ = np.zeros((n_windows, 5))
    for i, ind in enumerate(range(0, n_samples-n_taps+1, n_taps)):
        pow_[i] = (x[ind:ind+n_taps, :]**2).mean(0)
    powers[experiment] = pow_



import pandas as pd
from scipy.stats import ttest_ind, levene
results_df = pd.DataFrame()
tests = pd.DataFrame()
protocols = ['background', 'before-right', 'after-right', 'before-left', 'after-left']
for i_exp, experiment in enumerate(experiments):
    for i_protocol in range(5):
        pows = powers[experiments[i_exp]][:, i_protocol]
        results_df = results_df.append(pd.DataFrame({'subj': experiment.split('_')[1],
                        'protocol': protocols[i_protocol],
                        'power': pows}))
        if i_protocol >= 0:
            # t, p = levene(powers[experiments[i_exp]][:, 0], pows)
            t, p = ttest_ind(powers[experiments[i_exp]][:, 0], pows, equal_var=False)
            tests = tests.append(pd.DataFrame({'subj': experiment.split('_')[1],
                                               't-stat': [t], 'p-value': [p], 'protocol': protocols[i_protocol]}))

ax = sns.boxplot(x="protocol", y="power", hue="subj", data=results_df)
tests.to_csv('circle_border_tests' + '.csv')
plt.savefig('circle_border.png', dpi=200)
print(tests)



plt.show()