from collections import namedtuple

from pynfb.serializers.xml_ import get_lsl_info_from_xml
import numpy as np
import pylab as plt
from scipy.signal import hilbert, firwin2, filtfilt
from scipy.fftpack import rfft, irfft, fftfreq
import pandas as pd
import seaborn as sns

import pickle
import h5py




pilot_dir = 'D:\\Mu'
experiments = ['Dasha1_02-20_09-01-29',
               'Dasha2_02-22_15-53-52',
               'Dasha3_02-23_14-21-42',
               'Dasha4_02-24_16-59-08'][1:]

def fft_filter(x, fs, band=(9, 14)):
    w = fftfreq(x.shape[0], d=1. / fs * 2)
    f_signal = rfft(x)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(w < band[0]) | (w > band[1])] = 0
    cut_signal = irfft(cut_f_signal)
    return cut_signal

def filtfilt_filter(x, fs, band=(9, 14)):
    w = 0.
    taps = firwin2(2000, [0, band[0]-w, band[0], band[1], band[1]+w, fs/2], [0, 0, 1, 1, 0, 0], nyq=fs/2)
    return filtfilt(taps, [1.], x)

def get_power(x, n_windows=None):
    env = (np.abs(hilbert(x))**2)[100:-100]
    if n_windows is None:
        return env
    n_samples = env.shape[0] // n_windows
    print(n_samples)
    pows = [env[k * n_samples:(k + 1) * n_samples].mean() for k in range(n_windows)]
    return np.array(pows)


def get_powers(x, fs, n_windows=10, band=(9, 14)):
    n_samples = x.shape[0] // n_windows
    print('n samples', n_samples)
    w = fftfreq(x.shape[0], d=1. / fs * 2)
    f_signal = rfft(x)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(w < band[0]) | (w > band[1])] = 0
    cut_signal = irfft(cut_f_signal)
    plt.plot(cut_signal)
    plt.show()
    amps = np.zeros((n_windows, ))
    for k in range(n_windows):
        amps[k] = cut_signal[k*n_samples:(k+1)*n_samples].std()  # TODO: in one row
    return amps

results = {}

drop_channels = ['M_left', 'M_right']
pilot_dir = 'C:\\Users\\Nikolai\\Downloads'

subjs = [
    ['pilot_5Days_Rakhmankulov_Day1_02-27_17-27-34',
     'pilot5days_Rakhmankulov_Day2_02-28_14-45-36',
     'pilot5days_Rakhmankulov_Day3_03-01_12-51-41'],

     ['pilot_Cherlenok_Day1_02-27_12-51-56',
     'pilot5days_Cherlenok_Day2_02-28_15-50-08',
     'pilot5days_Cherlenok_Day3_03-01_16-24-03'],

    ['pilot5Days_Plackhin_Day1_02-27_16-04-08',
     'pilot5days_Plackhin_Day2_02-28_11-43-07',
     'pilot5days_Plackhin_Day3_03-01_11-45-35'],

    ['pilot5days_Skotnikova_Day1_02-27_15-15-18',
     'pilot5days_Skotnikova_Day2_02-28_14-06-40',
     'pilot5days_Skotnikova_Day3_03-01_10-44-28']
]

drop_channels = ['AUX', 'A1', 'A2']
experiments = subjs[0]

channel = 'C3'
reject_alpha = True
for experiment in experiments[:]:
    print('\n\nEXPERIMENT', experiment)

    # load rejections
    with h5py.File('{}\\{}\\{}'.format(pilot_dir, experiment, 'experiment_data.h5')) as f:
        rejections = [f['protocol1/signals_stats/left/rejections/rejection{}'.format(j + 1)][:]
                      for j in range(2)]
        rejection = rejections[0]
        if reject_alpha:
            rejection = np.dot(rejection, rejections[1])
        n_protocols = len([k for k in f.keys() if ('protocol' in k and k != 'protocol0')])
        print('number of protocols:', n_protocols)

    # load data
        labels, fs = get_lsl_info_from_xml(f['stream_info.xml'][0])
        print('fs: {}\nall labels {}: {}'.format(fs, len(labels), labels))
        channels = [label for label in labels if label not in drop_channels]
        print('selected channels {}: {}'.format(len(channels), channels))

        Record = namedtuple('Record', ['name', 'alpha_pow'])
        full_record = []
        slices = []
        data = []
        counter = 0
        for j in range(n_protocols):
            raw = f['protocol{}/raw_data'.format(j+1)][:]
            #raw = raw[:raw.shape[0] - raw.shape[0] % fs]

            x = np.dot(raw, rejection)[:, channels.index(channel)]
            x = dc_blocker(x)
            full_record.append(x)
            slices.append(slice(counter, counter + x.shape[0]))
            counter += x.shape[0]
        full_record = np.hstack(full_record)
        full_alpha = filtfilt_filter(full_record, fs, (9, 14))
        full_theta = filtfilt_filter(full_record, fs, (3, 6))
        #data1 = fft_filter(data, fs, (9, 14))
        if 1:

            cm = sns.color_palette()
            protocol_names = ['Filters', 'Rotate', 'Baseline', 'FB', 'Rest']
            c = dict(zip(protocol_names, [cm[j] for j in range(5)]))
            t = np.arange(full_record.shape[0]) / fs
            for j, slc in enumerate(slices):
                name = f['protocol{}'.format(j + 1)].attrs['name']
                plt.plot(t[slc], full_alpha[slc], c=c[name])
            plt.legend(protocol_names)
            plt.xlabel('time, s')
            plt.ylabel('alpha^2')
            plt.title(experiment)
            plt.show()

        theta = []
        for j in range(n_protocols):
            name = f['protocol{}'.format(j+1)].attrs['name']
            if name in ['FB']:
                print(len(full_alpha[slices[j]]))
                theta.append(get_power(full_theta[slices[j]])[0])
        theta = np.mean(theta)
        print(theta)

        for j in range(n_protocols):
            name = f['protocol{}'.format(j+1)].attrs['name']
            print(name)
            data.append(Record(
                name=name,
                alpha_pow=get_power(full_alpha[slices[j]]) / theta
            ))


        results[experiment] = data

import pandas as pd
from scipy.stats import ttest_ind, levene
results_df = pd.DataFrame()
tests = pd.DataFrame()
for i_exp, experiment in enumerate(experiments):
    for i_protocol, protocol in enumerate(results[experiment][:]):
        if protocol.name.upper() in ['ROTATE', 'FILTERS', 'FB']:
            results_df = results_df.append(pd.DataFrame({
                'exp': experiment,
                'protocol': '{}_{}({})'.format(i_exp + 1, protocol.name, i_protocol),
                'pow': protocol.alpha_pow[len(protocol.alpha_pow)//2:] if protocol.name == 'Filters' else protocol.alpha_pow}))

#ax = sns.violinplot(x="protocol", y="pow", hue="exp", data=results_df, )
sns.pointplot(x="protocol", y="pow", hue="exp", data=results_df)
#ax = sns.swarmplot(x="protocol", y="pow", data=results_df, color=".25")
sns.plt.xticks(rotation=30)


'''
rest = []
rot = []
for i_exp, experiment in enumerate(experiments):
    for i_protocol, protocol in enumerate(results[experiment][:-1]):
        if protocol.name.upper() == 'FILTERS':
            rest.append(protocol.alpha_pow[len(protocol.alpha_pow)//2:].mean())
        elif protocol.name.upper() == 'ROTATE':
            rot.append(protocol.alpha_pow.mean())
plt.show()
plt.stem(np.array(rot)/np.array(rest))
'''

plt.show()
