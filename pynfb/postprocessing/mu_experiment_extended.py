from collections import OrderedDict

import h5py
import numpy as np
import pylab as plt
import seaborn as sns
from mne.viz import plot_topomap

from pynfb.serializers.xml_ import get_lsl_info_from_xml
from pynfb.postprocessing.utils import fft_filter, load_rejections, get_info, get_protocol_power, get_colors, add_data
from pynfb.widgets.helpers import ch_names_to_2d_pos


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
            #plt.plot(fft_filter(f['protocol6/raw_data'][:, ch], fs, band=(3, 35)))
            #plt.plot(fft_filter(np.dot(f['protocol6/raw_data'], rejections)[:, ch], fs, band=(3, 35)))
            #plt.show()
            #from scipy.signal import welch
            #plt.plot(*welch(f['protocol1/raw_data'][:60*500//2, channels.index('C3')], fs, nperseg=1000))
            #plt.plot(*welch(f['protocol1/raw_data'][60*500//2:, channels.index('C3')], fs, nperseg=1000))

            #plt.plot(*welch(f['protocol2/raw_data'][:30*500//2, channels.index('C3')], fs, nperseg=1000))
            #plt.plot(*welch(f['protocol2/raw_data'][30*500//2:, channels.index('C3')], fs, nperseg=1000))
            #plt.legend(['Close', 'Open', 'Left', 'Right'])
            #plt.show()

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
            n_tops = top_ica.shape[1] + top_alpha.shape[1]
            for j_t in range(top_ica.shape[1]):
                ax = fg.add_subplot(4, n_tops * len(subj), n_tops * len(subj) * 3 + n_tops * j_s + j_t + 1)
                ax.set_xlabel('ICA{}'.format(j_t + 1))
                labels, fs = get_lsl_info_from_xml(f['stream_info.xml'][0])
                channels = [label for label in labels if label not in drop_channels]
                pos = ch_names_to_2d_pos(channels)
                plot_topomap(data=top_ica[:, j_t], pos=pos, axes=ax, show=False)
            for j_t in range(top_alpha.shape[1]):
                ax = fg.add_subplot(4, n_tops * len(subj),
                                    n_tops * len(subj) * 3 + n_tops * j_s + j_t + 1 + top_ica.shape[1])
                ax.set_xlabel('CSP{}'.format(j_t + 1))
                labels, fs = get_lsl_info_from_xml(f['stream_info.xml'][0])
                channels = [label for label in labels if label not in drop_channels]
                pos = ch_names_to_2d_pos(channels)
                plot_topomap(data=top_alpha[:, j_t], pos=pos, axes=ax, show=False)

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
            for j_p, ((name, pow), (name, x)) in enumerate(zip(powers.items(), raw.items())):
                if name == '2228. FB':
                    from scipy.signal import periodogram
                    fff = plt.figure()
                    fff.gca().plot(*periodogram(x, fs, nfft=fs * 3), c=cm[name.split()[1]])
                    plt.xlim(0, 80)
                    plt.ylim(0, 3e-11)
                    plt.show()
                print(name)
                time = np.arange(t, t + len(x)) / fs
                color = cm[''.join([i for i in name.split()[1] if not i.isdigit()])]
                ax1.plot(time, fft_filter(x, fs, (2, 45)), c=color, alpha=0.4)
                ax1.plot(time, alpha[name], c=color)
                t += len(x)
                ax.plot([j_p], [pow.mean() / norm], 'o', c=color, markersize=10)
                ax.errorbar([j_p], [pow.mean() / norm], yerr=pow.std() / norm, c=color, ecolor=color)
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
    settings_file = 'D:\\vnd_spbu\\pilot\\mu5days\\vnd_spbu_5days.json'
    #settings_file = 'D:\\vnd_spbu\\mock\\vnd_spbu_5days.json'
    with open(settings_file, 'r', encoding="utf-8") as f:
        settings = loads(f.read())

    channel = 'C3'
    reject_alpha = False
    normalize_by = 'beta'

    for j, subj in enumerate(settings['subjects'][4:]):
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