import numpy as np
import pylab as plt
import h5py
import mne

from scipy.signal import welch, hilbert
from pynfb.postprocessing.utils import get_info, add_data_simple, fft_filter, dc_blocker, load_rejections
from collections import OrderedDict
from IPython.display import clear_output

# load raw
from json import loads

from pynfb.protocols.ssd.topomap_selector_ica import ICADialog

settings_file = 'C:\_NFB\old_desctop\\kriv\\vnd_spbu_5days.json'
with open(settings_file, 'r') as f:
    settings = loads(f.read())

dir_ = settings['dir']
subj = 0
experiments = settings['subjects'][subj]
def preproc(x, fs, rej=None):
    #x = dc_blocker(x)
    #x = fft_filter(x, fs, band=(0, 45))
    if rej is not None:
        x = np.dot(x, rej)
    return x


#cm = get_colors2()
# plot spectrum
ch_plot = ['Fp1', 'C3', 'P3']
fig2, axes = plt.subplots(len(ch_plot), ncols=1, sharex=True, sharey=True)
import seaborn as sns
cm = sns.color_palette('Paired')


peaks = [50, 150]
ins = np.zeros(shape=(len(ch_plot), len(peaks)))
out = np.zeros(shape=(len(ch_plot), len(peaks)))

for j_experiment, experiment in enumerate(experiments):
    #experiment = experiments[0]




    reject = False
    with h5py.File('{}\\{}\\{}'.format(settings['dir'], experiment, 'experiment_data.h5')) as f:
        fs, channels, p_names = get_info(f, settings['drop_channels'])
        rejection, alpha, ica = None, None, None#load_rejections(f, reject_alpha=True)
        raw_before = OrderedDict()
        raw_after = OrderedDict()
        for j, name in enumerate(p_names):
            if j < 4:
                x = preproc(f['protocol{}/raw_data'.format(j + 1)][:], fs, rejection if reject else None)
                raw_before = add_data_simple(raw_before, name, x)





    for j, ch in enumerate(ch_plot):
        leg = []
        for key in ['Open']:#, 'Close', 'Left', 'Right']:
            #for x, style in zip([raw_before[key], raw_after[key]], ['', '--']):
                #f, Pxx = welch(x[:, channels.index(ch)], fs, nperseg=2048,)
                #axes[j].plot(f, Pxx, style,  c=cm[key])
                #x_plot = np.abs(hilbert(fft_filter(x[:, channels.index(ch)], fs)))
                #leg.append('P={:.3f}, D={:.2f}s'.format(Pxx[(f>9) & (f<14)].mean(), sum((x_plot > 5))/fs))

            f, Pxx = welch(raw_before[key][:, channels.index(ch)], fs, nperseg=2048, )
            for j_peak, peak in enumerate(peaks):
                val = max(Pxx[(f > peak-2.5) & (f < peak+2.5)])
                if j_experiment<2:
                    out[j, j_peak] += val
                else:
                    ins[j, j_peak] += val

            axes[j].semilogy(f, Pxx, alpha=1, c=cm[j_experiment+2])
            axes[j].set_xlim(0, 250)
            axes[j].set_ylim(1e-19, 5e-10)
            #x_plot = np.abs(hilbert(fft_filter(raw_before[key][:, channels.index(ch)], fs)))
            #leg.append('P={:.3f}, D={:.3f}s'.format(Pxx[(f > 9) & (f < 14)].mean(), sum((x_plot > 5)) / fs/2))
        #axes[j].legend(leg)
        axes[j].set_ylabel(ch)
        axes[j].grid(True)


ratio = out / ins
for j, ch in enumerate(ch_plot):
    for j_peak, peak in enumerate(peaks):
        axes[j].text(peak+2, out[j, j_peak]/2, '{}Hz: {:.0f}'.format(peak, ratio[j, j_peak]))

axes[-1].set_xlabel('Frequency, Hz')
axes[0].legend(['Outside (corner)', 'Outside (center)', 'Inside (opened door)', 'Inside (closed door)'])
plt.show()
