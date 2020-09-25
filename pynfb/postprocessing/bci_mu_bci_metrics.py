import numpy as np
import pylab as plt
import pandas as pd
from collections import OrderedDict
from json import loads
from mne.viz import plot_topomap
from pynfb.postprocessing.utils import get_info, add_data, get_colors_f, fft_filter, dc_blocker, load_rejections, \
    find_lag
from pynfb.protocols.ssd.topomap_selector_ica import ICADialog
from scipy.signal import welch, hilbert
import h5py
import pandas as pd
import seaborn as sns
from scipy import stats
from pynfb.signals.bci import BCISignal

dir_ = r'D:\bci_nfb_bci\bci_nfb_bci'
with open(dir_ + '\\info_mock.json', 'r', encoding="utf-8") as f:
    settings = loads(f.read())

def drop_outliers(x, threshold=0.00005, window=500):
    outliers_mask = np.diff((np.abs(x) > threshold).astype(int))
    good = np.ones_like(x)
    for k in np.where(outliers_mask>0)[0]:
        good[k:k+window] = 0
    return x[good>0]


def add_metrics(df, subj, day, after, slope, a_train, a_test):
    df.loc[len(metrics)] = {
        'subj': subj, 'day': day, 'after': after, 'fb_slope': slope,
        'acc_train': a_train[0], 'acc_train0': a_train[1], 'acc_train1': a_train[2], 'acc_train2': a_train[3],
        'acc_test': a_test[0], 'acc_test0': a_test[1], 'acc_test1': a_test[2], 'acc_test2': a_test[3],
    }
    return df

metrics = pd.DataFrame(columns=['subj', 'day', 'after', 'acc_train', 'acc_train0', 'acc_train1', 'acc_train2',
                                'acc_test', 'acc_test0', 'acc_test1', 'acc_test2', 'fb_slope'])
metrics = pd.read_csv('bcinfbbci_mock_metrics.csv')
print(metrics)
for subj, experiments in enumerate(settings['subjects']):
    for day, experiment in enumerate(experiments):

        print(day, subj, experiment)
        if subj > 4:
            with h5py.File('{}\\{}\\{}'.format(dir_, experiment, 'experiment_data.h5')) as f:
                fs, channels, p_names = get_info(f, settings['drop_channels'])
                spatial = f['protocol15/signals_stats/left/spatial_filter'][:]
                band = f['protocol15/signals_stats/left/bandpass'][:]
                print('band:', band)
                signals = OrderedDict()
                raw = OrderedDict()
                for j, name in enumerate(p_names):
                    if name != 'Bci':
                        x = np.dot(f['protocol{}/raw_data'.format(j + 1)][:], spatial)
                        x = drop_outliers(x)
                        x = fft_filter(x, fs, band=band)
                        #x[x > 0.00005] = np.nan
                        signals = add_data(signals, name, x, j)
                        raw = add_data(raw, name, f['protocol{}/raw_data'.format(j + 1)][:], j)


                norm_coeff = np.std(signals['16. Baseline'])
                fb_vars = []
                for fb_name in ['18. FB', '20. FB', '22. FB']:
                    for fb_part in np.split(signals[fb_name], range(0, len(signals[fb_name]), 2*fs))[1:-1]:
                        fb_vars.append(fb_part.std())
                fb_vars = np.array(fb_vars) / norm_coeff
                #sns.regplot(np.arange(len(fb_vars)), fb_vars)
                slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(fb_vars)), fb_vars)
                #slopes.append(slope)

                # get_accuracies
                # print('bci before:', list(raw.keys())[:9])
                bci = BCISignal(fs, channels, 'bci', 0)
                def get_Xy(protocols):
                    print(protocols)
                    X = [raw[prot] for prot in protocols]
                    def get_state(name):
                        if 'Open' in name:
                            return 0
                        elif 'Left' in name:
                            return 1
                        elif 'Right' in name:
                            return 2
                        else:
                            raise TypeError('Bad state', name)
                    y = [np.ones(len(raw[prot])) * get_state(prot) for prot in protocols]
                    X = np.vstack(X)
                    y = np.concatenate(y, 0)
                    return X, y

                X_before, y_before = get_Xy(list(raw.keys())[:12])
                #print(list(raw.keys())[:12])
                X_after, y_after = get_Xy(list(raw.keys())[-13:-1])
                #print(list(raw.keys())[-13:-1])
                for k in range(5):

                    bci.reset_model()
                    a_train = bci.fit_model(X_before, y_before)
                    a_test = bci.model.get_accuracies(X_after, y_after)
                    metrics = add_metrics(metrics, subj, day, 0, slope, a_train, a_test)
                    print('before-after', a_train, a_test)

                    bci.reset_model()
                    a_train = bci.fit_model(X_after, y_after)
                    a_test = bci.model.get_accuracies(X_before, y_before)
                    metrics = add_metrics(metrics, subj, day, 1, slope, a_train, a_test)
                    print('after-before', a_train, a_test)

                    # fill metrics: ['subj', 'day', 'before', 'acc_train', 'acc_train0', 'acc_train1', 'acc_train2',
                    #'acc_test', 'acc_test0', 'acc_test1', 'acc_test2', 'fb_slope']
                    print(metrics)

            metrics.to_csv('bcinfbbci_mock_metrics.csv', index=False)
        print('done')