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
    n_steps = int(n_sec * fs)
    w = fftfreq(n_steps, d=1. / fs * 2)
    #print(len(range(0, x.shape[0] - n_steps, n_steps)))
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
    filter_ = f['protocol1/signals_stats/left/spatial_filter'][:]
    rejection = np.dot(rejection, filter_)
    #alpha = filter_.reshape(-1, 1)
    return rejection, len(ica.T), np.hstack([ica, alpha])

def get_info(f, drop_channels):
    labels, fs = get_lsl_info_from_xml(f['stream_info.xml'][0])
    print('fs: {}\nall labels {}: {}'.format(fs, len(labels), labels))
    channels = [label for label in labels if label not in drop_channels]
    #print('selected channels {}: {}'.format(len(channels), channels))
    n_protocols = len([k for k in f.keys() if ('protocol' in k and k != 'protocol0')])
    protocol_names = [f['protocol{}'.format(j+1)].attrs['name'] for j in range(n_protocols)]
    #print('protocol_names:', protocol_names)
    return fs, channels, protocol_names

def get_protocol_power(f, i_protocol, fs, rejection, ch, band=(9, 14), dc=False):
    raw = f['protocol{}/raw_data'.format(i_protocol + 1)][:]
    x = np.dot(raw, rejection)#[:, ch]
    if dc:
        x = dc_blocker(x)
    return get_power2(x, fs, band), fft_filter(x, fs, band), x

def get_colors():
    p_names = [ 'Right', 'Left',  'Rest', 'FB', 'Closed', 'Opened', 'Baseline']
    cm = sns.color_palette('Paired', n_colors=len(p_names))
    c = dict(zip(p_names, [cm[j] for j in range(len(p_names))]))
    return c


def add_data(powers, name, pow, j):
    if name == 'Filters':
        powers['{}. Closed'.format(j + 1)] = pow[:len(pow) // 2]
        powers['{}. Opened'.format(j + 1)] = pow[len(pow) // 2:]
    elif name == 'Rotate':
        powers['{}. Right'.format(j + 1)] = pow[:len(pow) // 2]
        powers['{}. Left'.format(j + 1)] = pow[len(pow) // 2:]
    elif 'FB' in name and len(name)>2:
        powers['{}. FB'.format(j + 1)] = pow
    else:
        powers['{}. {}'.format(j + 1, name)] = pow
    return powers


def plot_results(pilot_dir, subj, channel, alpha_band=(9, 14), theta_band=(3, 6), drop_channels=None, dc=False,
                 reject_alpha=True, normalize_by='opened'):
    drop_channels = drop_channels or []
    cm = get_colors()
    fg = plt.figure(figsize=(30, 6))
    for j_s, experiment in enumerate(subj):
        with h5py.File('{}\\{}\\{}'.format(pilot_dir, experiment, 'experiment_data.h5')) as f:
            rejections, top_alpha, top_ica = load_rejections(f, reject_alpha=reject_alpha)
            fs, channels, p_names = get_info(f, drop_channels)
            ch = channels.index(channel)

            # collect powers
            powers = OrderedDict()
            raw = OrderedDict()
            alpha = OrderedDict()
            pow_theta = []
            for j, name in enumerate(p_names):
                pow, alpha_x, x = get_protocol_power(f, j, fs, rejections, ch, alpha_band, dc=dc)
                if 'FB' in name:
                    pow_theta.append(get_protocol_power(f, j, fs, rejections, ch, theta_band, dc=dc)[0].mean())
                powers = add_data(powers, name, pow, j)
                raw = add_data(raw, name, x, j)
                alpha = add_data(alpha, name, alpha_x, j)

            # plot rejections
            n_tops = top_ica.shape[1] #+ top_alpha.shape[1]
            for j_t in range(top_ica.shape[1]):
                ax = fg.add_subplot(4, n_tops * len(subj), n_tops * len(subj) * 3 + n_tops * j_s + j_t + 1)
                ax.set_xlabel('ICA{}'.format(j_t + 1) if j_t < top_alpha else 'CSP{}'.format(-top_alpha + j_t + 1))
                labels, fs = get_lsl_info_from_xml(f['stream_info.xml'][0])
                channels = [label for label in labels if label not in drop_channels]
                pos = ch_names_to_2d_pos(channels)
                plot_topomap(data=top_ica[:, j_t], pos=pos, axes=ax, show=False)


            # plot powers
            if normalize_by == 'opened':
                norm = powers['1. Opened'].mean()
            elif normalize_by == 'beta':
                norm = np.mean(pow_theta)
            else:
                print('WARNING: norm = 1')
            print('norm', norm)

            ax1 = fg.add_subplot(3, len(subj), j_s + 1)
            ax = fg.add_subplot(3, len(subj), j_s + len(subj) + 1)
            t = 0
            #print(powers.keys())
            if j_s == 0:
                print(powers['4. FB'].mean() / powers['3. Baseline'].mean())
            for j_p, ((name, pow), (name, x)) in enumerate(zip(powers.items(), raw.items())):
                if name == '2228. FB':
                    from scipy.signal import periodogram
                    fff = plt.figure()
                    fff.gca().plot(*periodogram(x, fs, nfft=fs * 3), c=cm[name.split()[1]])
                    plt.xlim(0, 80)
                    plt.ylim(0, 3e-11)
                    plt.show()
                #print(name)
                time = np.arange(t, t + len(x)) / fs
                ax1.plot(time, x, c=cm[name.split()[1]], alpha=0.4)
                ax1.plot(time, alpha[name], c=cm[name.split()[1]])
                t += len(x)
                ax.plot([j_p], [pow.mean() / norm], 'o', c=cm[name.split()[1]], markersize=10)
                c = cm[name.split()[1]]
                ax.errorbar([j_p], [pow.mean() / norm], yerr=pow.std() / norm, c=c, ecolor=c)
            fb_x = np.hstack([[j] * len(pows) for j, (key, pows) in enumerate(powers.items()) if 'FB' in key])
            fb_y = np.hstack([pows for key, pows in powers.items() if 'FB' in key]) / norm
            sns.regplot(x=fb_x, y=fb_y, ax=ax, color=cm['FB'], scatter=False, truncate=True)

            ax1.set_xlim(0, t / fs)
            ax1.set_ylim(-40, 40)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
            ax.set_xticks(range(len(powers)))
            ax.set_xticklabels(powers.keys())
            ax.set_ylim(0, 3)
            ax.set_xlim(-1, len(powers))
            ax1.set_title('Day {}'.format(j_s + 1))
    return fg


if __name__ == '__main__':

    from json import loads
    settings_file = 'D:\\vnd_spbu\\mock\\vnd_spbu_5days.json'
    with open(settings_file, 'r', encoding="utf-8") as f:
        settings = loads(f.read())

    channel = 'C3'
    reject_alpha = True
    normalize_by = 'beta'

    for j, subj in enumerate(settings['subjects']):
        #pass

        fg = plot_results(settings['dir'], subj,
                     channel=channel,
                     alpha_band=(9, 14),
                     theta_band=(3, 6),
                     drop_channels=settings['drop_channels'],
                     dc=True,
                     reject_alpha=reject_alpha,
                     normalize_by=normalize_by)

        fg.savefig('S{s}_ch{ch}_{csp}normby_{norm}.png'.format(
            s=j, ch=channel, csp='csp_' if reject_alpha else '', norm=normalize_by),
            dpi=300)
        plt.show()