from collections import OrderedDict

import numpy as np
import pylab as plt
import h5py
from mne.viz import plot_topomap
from scipy.signal import hilbert, firwin2, filtfilt
from scipy.fftpack import rfft, irfft, fftfreq

from pynfb.serializers.xml_ import get_lsl_info_from_xml
from pynfb.signals.rejections import Rejections
from pynfb.signal_processing.filters import SpatialRejection
import seaborn as sns

from pynfb.widgets.helpers import ch_names_to_2d_pos


def dc_blocker(x, r=0.99):
    # DC Blocker https://ccrma.stanford.edu/~jos/fp/DC_Blocker.html
    y = np.zeros_like(x)
    for n in range(1, x.shape[0]):
        y[n] = x[n] - x[n-1] + r * y[n-1]
    return y

def fft_filter(x, fs, band=(9, 14)):
    w = fftfreq(x.shape[0], d=1. / fs * 2)
    f_signal = rfft(x, axis=0)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(w < band[0]) | (w > band[1])] = 0
    cut_signal = irfft(cut_f_signal, axis=0)
    return cut_signal


def get_power2(x, fs, band, n_sec=5):
    n_steps = n_sec * fs
    w = fftfreq(n_steps, d=1. / fs * 2)
    print(len(range(0, x.shape[0] - n_steps, n_steps)))
    pows = [2*np.sum(rfft(x[k:k+n_steps])[(w > band[0]) & (w < band[1])]**2)/n_steps
            for k in range(0, x.shape[0] - n_steps, n_steps)]
    return np.array(pows)

def get_power(x, fs, band):
    #w = 0.
    #taps = firwin2(1000, [0, band[0]-w, band[0], band[1], band[1]+w, fs/2], [0, 0, 1, 1, 0, 0], nyq=fs/2)
    #x = filtfilt(taps, [1.], x)
    x = fft_filter(x, fs, band)
    return x**2

def load_rejections(f, reject_alpha=True):
    rejections = [f['protocol1/signals_stats/left/rejections/rejection{}'.format(j + 1)][:] for j in range(2)]
    alpha = f['protocol1/signals_stats/left/rejections/rejection2_topographies'][:]
    ica = f['protocol1/signals_stats/left/rejections/rejection1_topographies'][:]
    rejection = rejections[0]
    if reject_alpha:
        rejection = np.dot(rejection, rejections[1])
    return rejection, alpha, ica

def get_info(f, drop_channels):
    labels, fs = get_lsl_info_from_xml(f['stream_info.xml'][0])
    print('fs: {}\nall labels {}: {}'.format(fs, len(labels), labels))
    channels = [label for label in labels if label not in drop_channels]
    print('selected channels {}: {}'.format(len(channels), channels))
    n_protocols = len([k for k in f.keys() if ('protocol' in k and k != 'protocol0')])
    protocol_names = [f['protocol{}'.format(j+1)].attrs['name'] for j in range(n_protocols)]
    print('protocol_names:', protocol_names)
    return fs, channels, protocol_names

def get_protocol_power(f, i_protocol, fs, rejection, ch, band=(9, 14), dc=False):
    raw = f['protocol{}/raw_data'.format(i_protocol + 1)][:]
    x = np.dot(raw, rejection)[:, ch]
    if dc:
        x = dc_blocker(x)
    return get_power2(x, fs, band)

def get_colors():
    p_names = [ 'Right', 'Left',  'Rest', 'FB', 'Closed', 'Opened', 'Baseline']
    cm = sns.color_palette('Paired', n_colors=len(p_names))
    c = dict(zip(p_names, [cm[j] for j in range(len(p_names))]))
    return c


if __name__ == '__main__':
    import h5py

    pilot_dir = ['C:\\Users\\Nikolai\\Downloads', 'D:\\Mu'][1]
    subjs = [
        ['pilot_5Days_Rakhmankulov_Day1_02-27_17-27-34',
         'pilot5days_Rakhmankulov_Day2_02-28_14-45-36',
         'pilot5days_Rakhmankulov_Day3_03-01_12-51-41',
         'pilot5days_Rakhmankulov_Day4_03-02_17-26-28',
         'pilot5days_Rakhmankulov_Day5_03-04_09-24-16'],

        ['pilot_Cherlenok_Day1_02-27_12-51-56',
         'pilot5days_Cherlenok_Day2_02-28_15-50-08',
         'pilot5days_Cherlenok_Day3_03-01_16-24-03',
         'pilot5days_Cherlenok_Day4_03-02_16-45-43',
         'pilot5days_Cherlenok_Day5_03-03_12-35-46'],

        ['pilot5Days_Plackhin_Day1_02-27_16-04-08',
         'pilot5days_Plackhin_Day2_02-28_11-43-07',
         'pilot5days_Plackhin_Day3_03-01_11-45-35',
         'pilot5days_Plackhin_Day4_03-02_11-20-43',
         'pilot5days_Plackhin_Day5_03-03_14-29-52'],

        ['pilot5days_Skotnikova_Day1_02-27_15-15-18',
         'pilot5days_Skotnikova_Day2_02-28_14-06-40',
         'pilot5days_Skotnikova_Day3_03-01_10-44-28',
         'pilot5days_Skotnikova_Day4_03-02_13-33-55',
         'pilot5days_Skotnikova_Day5_03-03_13-42-04'],

        ['Dasha1_02-20_09-01-29',
        'Dasha2_02-22_15-53-52',
        'Dasha3_02-23_14-21-42',
        'Dasha4_02-24_16-59-08'],
        ['Andrey1_03-07_19-05-02']][-1:]
    drop_channels = [['AUX', 'A1', 'A2'], ['M_left', 'M_right']][1]
    channel = 'C4'
    alpha_band = (9, 14)
    theta_band = (3, 6)
    cm = get_colors()

    for subj in subjs[:]:
        fg = plt.figure(figsize=(30, 6))
        for j_s, experiment in enumerate(subj):
            with h5py.File('{}\\{}\\{}'.format(pilot_dir, experiment, 'experiment_data.h5')) as f:
                rejections, top_alpha, top_ica = load_rejections(f, reject_alpha=True)
                fs, channels, p_names = get_info(f, drop_channels)
                ch = channels.index(channel)

                # collect powers
                powers = OrderedDict()
                pow_theta = []
                for j, name in enumerate(p_names):
                    pow = get_protocol_power(f, j, fs, rejections, ch, alpha_band, dc=True)
                    if name == 'FB':
                        pow_theta.append(get_protocol_power(f, j, fs, rejections, ch, theta_band, dc=True).mean())
                    if name == 'Filters':
                        powers['{}. Closed'.format(j+1)] = pow[:len(pow)//2]
                        powers['{}. Opened'.format(j+1)] = pow[len(pow)//2:]
                    elif name == 'Rotate':
                        powers['{}. Right'.format(j+1)] = pow[:len(pow)//2]
                        powers['{}. Left'.format(j+1)] = pow[len(pow)//2:]
                    else:
                        powers['{}. {}'.format(j+1, name)] = pow


                # plot rejections
                for j_t in range(top_ica.shape[1]):
                    ax = fg.add_subplot(5, top_ica.shape[1]*len(subj), top_ica.shape[1]*len(subj)*3 + top_ica.shape[1]*j_s + j_t + 1)
                    ax.set_xlabel('ICA{}'.format(j_t+1))
                    labels, fs = get_lsl_info_from_xml(f['stream_info.xml'][0])
                    channels = [label for label in labels if label not in drop_channels]
                    pos = ch_names_to_2d_pos(channels)
                    plot_topomap(data=top_ica[:, j_t], pos=pos, axes=ax, show=False)
                for j_t in range(top_alpha.shape[1]):
                    ax = fg.add_subplot(5, top_alpha.shape[1]*len(subj), top_alpha.shape[1]*len(subj)*4 + top_alpha.shape[1]*j_s + j_t + 1)
                    ax.set_xlabel('CSP{}'.format(j_t+1))
                    labels, fs = get_lsl_info_from_xml(f['stream_info.xml'][0])
                    channels = [label for label in labels if label not in drop_channels]
                    pos = ch_names_to_2d_pos(channels)
                    plot_topomap(data=top_alpha[:, j_t], pos=pos, axes=ax, show=False)


                # plot powers
                norm = powers['{}. Baseline'.format(p_names.index('Baseline') + 1)].mean()
                #norm = np.mean(pow_theta)
                print('norm', norm)
                ax = fg.add_subplot(2, len(subj), j_s + 1)
                for j_p, (name, pow) in enumerate(powers.items()):
                    print(name)
                    ax.plot([j_p], [pow.mean()/norm], 'o', c=cm[name.split()[1]], markersize=10)
                    c = cm[name.split()[1]]
                    ax.errorbar([j_p], [pow.mean()/norm], yerr=pow.std()/norm,  c=c, ecolor=c)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
                ax.set_xticks(range(len(powers)))
                ax.set_xticklabels(powers.keys())
                ax.set_ylim(0, 2)
                ax.set_xlim(-1, len(powers))
                ax.set_title('Day {}'.format(j_s+1))



        fg.savefig(' '.join(subj[-1].split('_')[:2])+channel+'.png', dpi=300)
        plt.show()