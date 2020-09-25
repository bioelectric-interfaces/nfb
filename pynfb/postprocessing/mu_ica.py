import h5py
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


# load raw
from pynfb.widgets.helpers import ch_names_to_2d_pos

mock = False


dir_ = 'D:\mu_ica\mu_ica'
with open(dir_ + '\\info.json', 'r', encoding="utf-8") as f:
    settings = loads(f.read())

subj = 0
day = 0


run_ica = False
reject = False
#for subj in range(4):
    #for day in range(5):
experiments = settings['subjects'][subj]
experiment = experiments[day]
for subj, experiments in enumerate(settings['subjects']):
    for day, experiment in enumerate(experiments):
        print(subj, day, experiment)
        def preproc(x, fs, rej=None):
            x = dc_blocker(x)
            x = fft_filter(x, fs, band=(0, 70))
            if rej is not None:
                x = np.dot(x, rej)
            return x


        def compute_lengths(x, gap, minimum):
            lengths = []
            x_copy = -x.astype(int).copy()+1
            c = 0
            for j, y in enumerate(x_copy):
                if y:
                    c += 1
                elif c > 0:
                    if c > gap:
                        lengths.append(c)
                    else:
                        x_copy[j-c:j] = 0
                    c = 0
            if len(lengths) == 0:
                lengths = [0]
            if minimum is None:
                #print(np.array(lengths)/500)
                return np.array(lengths), x_copy
            else:
                return compute_lengths(x_copy.copy(), minimum, None)

        def compute_heights(x, mask):
            lengths = []
            mask_copy = mask.astype(int).copy()
            lengths_buffer = []
            for j, y in enumerate(mask_copy):
                if y:
                    lengths_buffer.append(x[j])
                elif len(lengths_buffer) > 0:
                    lengths.append(np.mean(lengths_buffer))
                    lengths_buffer = []
            print(lengths)
            return np.array(lengths)



        with h5py.File('{}\\{}\\{}'.format(dir_, experiment, 'experiment_data.h5')) as f:
            fs, channels, p_names = get_info(f, settings['drop_channels'])
            if reject:
                rejections = load_rejections(f, reject_alpha=True)[0]
            else:
                rejections = None
            spatial = f['protocol15/signals_stats/left/spatial_filter'][:]
            mu_band = f['protocol15/signals_stats/left/bandpass'][:]
            #mu_band = (12, 13)
            max_gap = 1 / min(mu_band) * 2
            min_sate_duration = max_gap * 2
            raw = OrderedDict()
            signal = OrderedDict()
            for j, name in enumerate(p_names):
                x = preproc(f['protocol{}/raw_data'.format(j + 1)][:], fs, rejections)
                raw = add_data(raw, name, x, j)
                signal = add_data(signal, name, f['protocol{}/signals_data'.format(j + 1)][:], j)

        del raw[list(raw.keys())[-1]]
        # make csp:
        if run_ica:
            from PyQt5.QtWidgets import QApplication
            ap = QApplication([])
            all_keys = [key for key in raw.keys() if 'Left' in key or 'Right' in key or 'Close' in key or 'Open' in key]
            first = all_keys[:len(all_keys)//2]
            last = all_keys[len(all_keys)//2:]
            tops = []
            spats = []

            fig1, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False)
            for j, data in enumerate([first, last]):
                raw_data = np.concatenate([raw[key] for key in data])
                rej, spatial, top = ICADialog.get_rejection(raw_data, channels, fs, mode='ica', states=None)[:3]
                tops.append(top)
                spats.append(spatial)
                plot_topomap(top, ch_names_to_2d_pos(channels), axes=axes[j, 0], show=False)
                plot_topomap(spatial, ch_names_to_2d_pos(channels), axes=axes[j, 1], show=False)
                axes[j, 0].set_xlabel('Topography ({})'.format('before' if j == 0 else 'after'))
                axes[j, 1].set_xlabel('Spatial filter ({})'.format('before' if j == 0 else 'after'))
            #plt.show()



        # plot raw data
        ch_plot = ['C3', 'P3', 'ICA']#, 'Pz', 'Fp1']
        fig1, axes = plt.subplots(len(ch_plot), ncols=1, sharex=True, sharey=False)
        print(axes)

        #find median
        x_all = []
        x_all_ica = []
        for name, x in raw.items():
            if 'Baseline' in name:
                x_all.append(np.abs(hilbert(fft_filter(x, fs, band=mu_band))))
                x_all_ica.append(np.abs(hilbert(np.dot(fft_filter(x, fs, band=mu_band), spatial))))
                break
        x_median = np.median(np.concatenate(x_all), 0)
        x_f_median = np.median(np.concatenate(x_all_ica))


        coef = 1
        # plot raw
        t = 0
        cm = get_colors_f
        fff = plt.figure()
        for name, x in list(raw.items()):
            for j, ch in enumerate(ch_plot):
                time = np.arange(t, t + len(x)) / fs
                y = x[:, channels.index(ch)] if ch != 'ICA' else np.dot(x, spatial)
                x_plot = fft_filter(y, fs, band=(2, 45))
                axes[j].plot(time, x_plot, c=cm(name), alpha=1)
                envelope = np.abs(hilbert(fft_filter(y, fs, band=mu_band)))

                #axes[j].plot(time, envelope, c=cm(name), alpha=1, linewidth=1)

                #axes[j].plot(time, signal[name][:, 0]* envelope.std() + envelope.mean(), c='k', alpha=1)
                threshold = coef*x_median[channels.index(ch)] if ch != 'ICA' else coef*x_f_median
                sc = 10*envelope.mean()

                lengths, x_copy = compute_lengths(envelope > threshold, fs*max_gap, fs*min_sate_duration)
                #axes[j].fill_between(time, -x_copy * sc*0, (envelope > threshold)*sc, facecolor=cm(name), alpha=0.6, linewidth=0)
                axes[j].fill_between(time, -x_copy * sc * 0 - 1, +(x_copy)*sc * 0 + 1, facecolor=cm(name), alpha=1, linewidth=0)
                axes[j].set_ylabel(ch)
                axes[j].set_xlabel('time, [s]')
                #axes[j].set_ylim(-1e-4, 1e-4)
                #axes[j].set_xlim(190, 205)
            t += len(x)
        axes[0].set_xlim(0, t/fs)
        axes[0].set_title('Day {}'.format(day+1))

        keys = [key for key in raw.keys() if 'FB' in key or 'Baseline' in key]
        # plot spectrum
        ch_plot = ['ICA']
        fig2, axes = plt.subplots(len(ch_plot), ncols=1, sharex=True, sharey=False, figsize=(8,3))
        axes = [axes]
        y_max = [0.4e-10, 2.5, 10, 20, 20][subj]
        for j, ch in enumerate(ch_plot):
            leg = []
            fb_counter = 0
            for jj, key in enumerate(keys):
                x = raw[key]
                style = '--' if fb_counter > 0 else ''
                w = 2
                if 'FB' in key:
                    fb_counter +=1
                    style = ''
                    w = fb_counter
                y = x[:, channels.index(ch)] if ch != 'ICA' else np.dot(x, spatial)
                f, Pxx = welch(y, fs, nperseg=2048,)
                axes[j].plot(f, Pxx, style,  c=cm(key), linewidth=w, alpha=0.8 if 'FB' in key else 1)
                x_plot = np.abs(hilbert(fft_filter(y, fs, band=mu_band)))
                threshold = coef * x_median[channels.index(ch)] if ch != 'ICA' else coef * x_f_median
                leg.append('{}'.format(key))


            axes[j].set_xlim(7, 15)
            axes[j].axvline(x=mu_band[0], color='k', alpha=0.5)
            axes[j].axvline(x=mu_band[1], color='k', alpha=0.5)
            #axes[j].set_ylim(0, 1e-10)
            axes[j].set_ylabel(ch)
            axes[j].set_ylabel('Hz')
            axes[j].legend(leg, loc='upper left')
        axes[0].set_title('S{} Day{} Band: {}-{} Hz'.format(subj, day+1, *mu_band))
        fig2.savefig('FBSpec_S{}_D{}'.format(subj, day + 1))
        plt.savefig('S{}_Day{}_spec'.format(subj, day+1))


        # desync
        print(raw.keys())
        baseline_keys = ['14. Baseline', '27. Baseline']
        motor_keys    = ['15. Left', '28. Left']
        fb_keys = [key for key in raw.keys() if 'FB' in key]
        #print()

        def get_mean_envelope(x):
            y = np.dot(x, spatial)
            envelope = np.abs(hilbert(fft_filter(y, fs, mu_band)))
            n_samples = len(envelope) // 10
            return [envelope[k * n_samples: (k+1) * n_samples].mean() for k in range(10)]

        #desync_before = get_mean_envelope(raw[motor_keys[0]]) / get_mean_envelope(raw[baseline_keys[0]])
        #desync_after  = get_mean_envelope(raw[motor_keys[1]]) / get_mean_envelope(raw[baseline_keys[1]])
        #import pandas as pd
        df = dict([('FB{}'.format(j+1), get_mean_envelope(raw[key])) for j, key in enumerate(fb_keys)])
        pd.DataFrame(df).to_csv('FB_{}.csv'.format(experiment))

        # print('Desync:', get_mean_envelope(raw[motor_keys[0]]), desync_after)



        # plot durations
        fig3, axes = plt.subplots(nrows=6, sharex=True, figsize=(4, 8))
        print(raw.keys())
        keys = ['14. Baseline', '17. FB', '19. FB', '21. FB', '23. FB', '25. FB', '27. Baseline']
        keys = [key for key in raw.keys() if 'FB' in key or 'Baseline' in key]

        dots = np.zeros((6, len(keys)))
        for jj, key in enumerate(keys):
            y = np.dot(raw[key], spatial)
            f, Pxx = welch(y, fs, nperseg=2048, )
            envelope = np.abs(hilbert(fft_filter(y, fs, mu_band)))
            threshold = coef * x_f_median
            lengths, x_copy = compute_lengths(envelope > threshold, fs*max_gap, fs*min_sate_duration)
            heights = compute_heights(envelope, x_copy)
            dots[0, jj] = envelope.mean()
            dots[1, jj] = sum(x_copy) / (len(envelope)) * 100
            dots[2, jj] = len(lengths)/(len(envelope)/fs/60)
            dots[3, jj] = lengths.mean()/fs
            dots[4, jj] = heights.mean()
            dots[5, jj] = find_lag(signal[key][:,0], envelope, fs, show=False)/fs
            for k in range(6):
                axes[k].plot([jj + 1], dots[k, jj], 'o', c=cm(key))

        import seaborn as sns
        for k in range(6):
            sns.regplot(np.arange(1, len(keys)+1), dots[k], ax=axes[k], color=cm('Baseline'),
                        line_kws={'alpha':0.4}, ci=None)
            sns.regplot(np.arange(1, len(keys)+1), dots[k], ax=axes[k], color=cm('FB'),
                        line_kws={'alpha':0.4}, ci=None)
            sns.regplot(np.arange(2, 7), dots[k,1:6], ax=axes[k], color=cm('FB'),
                        line_kws={'alpha': 0.7}, ci=None, truncate=True)
            if len(keys) == 7:
                axes[k].plot([1, 7], dots[k,[0,6]], c=cm('Baseline'), alpha=0.7, linewidth=3)

        titles = ['Mean envelope', 'Time in % for all mu-states', 'Number of mu-states per minute', 'Mean mu-state length [s]',
                  'Mean mu-state envelope', 'Mean lag']
        for ax, title in zip(axes, titles):
            ax.set_title(title)
            ax.set_xlim(0, len(keys)+1)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
            ax.set_xticks(range(1, len(keys)+1))
            ax.set_xticklabels(keys)
        plt.suptitle('S{} Day{} Band: {}-{} Hz'.format(subj, day+1, *mu_band))
        plt.savefig('S{}_Day{}_stats'.format(subj, day+1))
        #plt.show()
        plt.close()
