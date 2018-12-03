from pynfb.serializers.xml_ import get_lsl_info_from_xml
import numpy as np
import pylab as plt

from scipy.signal import *

import pandas as pd
import seaborn as sns
import pickle


pilot_dir = 'C:\\Users\\Nikolai\Downloads\pilot_5days'
pilot_dir = 'D:\\Mu'

experiments1 = ['pilot_Nikolay_1_10-17_13-57-56', #BAD NO FILTERS
                'pilot_Plackhin_1_10-20_12-03-01',
                'pilot_Tatiana_1_10-17_15-04-39',
                'pilot_Polyakova_1_10-24_15-21-18',
                'pilot_Emelyannikov29_1_11-09_20-14-28']

experiments2 = ['pilot_Nikolay_2_10-18_14-57-23',
                'pilot_Plackhin_2_10-21_13-09-27',
                'pilot_Tatiana_2_10-18_16-00-44',
                'pilot_Polyakova_2_10-25_14-19-56',
                'pilot_Emelyannikov29_2_11-10_19-47-25']

experiments = ['pilot_Skotnikova_Day1_01-31_12-55-31',
               'pilot_Skotnikova_Day2_02-01_14-32-24',
               'pilot_Skotnikova_Day3_02-02_15-36-52',
               'pilot_Skotnikova_Day4_02-03_15-57-31',
               'pilot_Skotnikova_Day5_02-04_10-45-13'][:5]


experiments = ['Dasha1_02-20_09-01-29',
               'Dasha2_02-22_15-53-52',
               'Dasha3_02-23_14-21-42',
               'Dasha4_02-24_16-59-08']

# experiments = experiments1 + experiments2
experiment_pairs = list(zip(experiments1, experiments2))

# experiments = experiment_pairs[0]
for _ in range(1):

    PROTOCOLS = {'FB': [3, 5, 7, 9, 11],
                 'BASELINE': 2,
                 'ROTATE': [1, 13],
                 'ALL': list(range(1, 15)),
                 'NAMES': ['Filters', 'Rotate', 'Baseline', 'FB', 'Rest', 'FB', 'Rest', 'FB', 'Rest', 'FB', 'Rest', 'FB',
                           'Rest', 'Rotate'],
                 'N_SAMPLES': [30000, 15000, 7500, 30000, 5000, 30000, 5000, 30000, 5000, 30000, 5000, 30000, 5000, 15000]}



    import h5py
    results = {}

    channel = 'C3'
    n_samples = 7500

    #new_rejections_file = 'new_rejections.pkl'
    #with open(new_rejections_file, 'rb') as handle:
    #    new_rejections = pickle.load(handle)

    use_pz = False
    reject_alpha = True
    for experiment in experiments[:]:
        print('\n\nEXPERIMENT', experiment)

        # load rejections:
        if use_pz:
            rejections = new_rejections[experiment]
        else:
            with h5py.File('{}\\{}\\{}'.format(pilot_dir, experiment, 'experiment_data.h5')) as f:
                rejections = [f['protocol1/signals_stats/left/rejections/rejection{}'.format(j + 1)][:]
                              for j in range(2)]

        rejection = rejections[0]
        if reject_alpha:
            rejection = np.dot(rejection, rejections[1])


        # load data
        with h5py.File('{}\\{}\\{}'.format(pilot_dir, experiment, 'experiment_data.h5')) as f:
            labels, fs = get_lsl_info_from_xml(f['stream_info.xml'][0])
            print('fs: {}\nall labels {}: {}'.format(fs, len(labels), labels))
            channels = [label for label in labels if label not in ['A1', 'A2', 'AUX']]
            pz_index = channels.index('Pz')
            print('selected channels {}: {}'.format(len(channels) - use_pz, channels))
            data = []
            for j in PROTOCOLS['ALL']:
                raw = f['protocol{}/raw_data'.format(j)][:]
                raw = raw[:raw.shape[0] - raw.shape[0] % fs]
                if use_pz:
                    raw = raw[:, np.arange(raw.shape[1]) != pz_index]
                data.append(np.dot(raw, rejection)[:, channels.index(channel)])
            assert [f['protocol{}'.format(j)].attrs['name'] for j in PROTOCOLS['ALL']] == PROTOCOLS['NAMES'], 'bad pr names'
            if 0:
                assert [dt.shape[0] for dt in data] == PROTOCOLS['N_SAMPLES'], \
                'bad samples number for {}:\nexpected\n{},\nget\n{}'.format(experiment, PROTOCOLS['N_SAMPLES'],
                                                                        [dt.shape[0] for dt in data])

            results[experiment] = data

    # collect scalers (beta)
    powers_beta = {}
    b, a = butter(4, [3 / fs * 2, 6 / fs * 2], 'band')
    for experiment in experiments:
        print('****', results[experiment][PROTOCOLS['BASELINE']].shape)
        plt.plot(filtfilt(b, a, results[experiment][PROTOCOLS['BASELINE']], axis=0))
        powers_beta[experiment] = filtfilt(b, a, results[experiment][PROTOCOLS['BASELINE']], axis=0).std()
        print('****', powers_beta)
    plt.show()


    # filter,  normalization and plot
    b, a = butter(3, [9 / fs * 2, 14 / fs * 2], 'band')
    f = plt.figure()
    n_plots = len(PROTOCOLS['ALL'])
    axs = [f.add_subplot(n_plots, 1, k+1) for k in range(n_plots)]
    for experiment in experiments[:]:
        # FILTER
        x = [filtfilt(b, a, x_, axis=0) if len(x_)>0 else x_ for x_ in results[experiment]]

        # NORMALIZATION
        x = [(x_ - np.mean(x[PROTOCOLS['BASELINE']], axis=0)) / (1 + 0 * x[PROTOCOLS['BASELINE']].std()) for x_ in x]
        results[experiment] = x

        # PLOT
        for k in range(n_plots):
            axs[k].plot(x[k])
            axs[k].set_xlim(0, 30000)
            axs[k].set_ylim(-4, 4)
    plt.title('{} {} 3-6Hz'.format(experiments[0].split('_')[1], channel))
    plt.legend(['{}st day'.format(k+1) for k, _e in enumerate(experiments)])
    plt.savefig('_'.join(experiments[0].split('_')[:2]) + channel + '.png', dpi=200)
    plt.show()





    # COLLECT POWERS
    n_windows = 10
    powers = dict([(experiment, np.zeros((len(PROTOCOLS['ALL']), n_windows))) for experiment in experiments])


    for experiment in experiments:
        for j, protocol_ind in enumerate(PROTOCOLS['ALL']):
            x = results[experiment][j] / powers_beta[experiment]
            pow_ = np.zeros((n_windows, ))
            inds, n_taps = np.linspace(0, len(x), n_windows+1, retstep=True, dtype=int)
            n_taps = int(n_taps)
            print(type(inds[0]), type(n_taps))
            for i, ind in enumerate(inds[:-1]):
                print(ind, n_taps)
                powers[experiment][j, i] = (x[ind:ind+n_taps]**2).mean(0)

    print(powers)

    import pandas as pd
    from scipy.stats import ttest_ind, levene
    results_df = pd.DataFrame()
    tests = pd.DataFrame()
    for i_exp, experiment in enumerate(experiments):
        protocols = ['{}_{}({})'.format(i_exp + 1, prot, j) for j, prot in enumerate(PROTOCOLS['NAMES'])]
        for i_protocol, protocol in enumerate(protocols):
            if i_protocol in PROTOCOLS['FB'] + [PROTOCOLS['BASELINE']]:
                pows = powers[experiments[i_exp]][i_protocol]
                name = '{} (day {})'.format(experiment.split('_')[1], i_exp+1)
                if i_protocol != PROTOCOLS['BASELINE']:
                    results_df = results_df.append(pd.DataFrame({'subj': name,
                                    'protocol': protocol,
                                    'power/power(3-6Hz)': pows}))

                    # t, p = levene(powers[experiments[i_exp]][:, 0], pows)
                    t, p = ttest_ind(powers[experiments[i_exp]][PROTOCOLS['BASELINE']], pows, equal_var=False)
                    tests = tests.append(pd.DataFrame({'subj': '_'.join(experiment.split('_')[1:3]),
                                                       't-stat': [t], 'p-value': [p], 'protocol': protocol}))

    ax = sns.boxplot(x="protocol", y="power/power(3-6Hz)", hue="subj", data=results_df)
    sns.plt.xticks(rotation=30)
    tests.to_csv('circle_border_tests' + '.csv')
    plt.ylim(0, 4)
    plt.title('{} {}'.format(experiments[0].split('_')[1], channel))
    plt.tight_layout()
    plt.savefig('_'.join(experiments[0].split('_')[:2]) + channel + '.png', dpi=200)

    print(tests)



    plt.show()