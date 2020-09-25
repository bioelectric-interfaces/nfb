import numpy as np
import pylab as plt
import h5py

import mne

from json import loads

from mne.channels import make_eeg_layout
from mne.viz import iter_topography


def add_data_simple(odict, name, x, info):
    to_raw = lambda y: mne.io.RawArray(y.T, info)
    if name == 'Filters':
        odict['Closed'] = to_raw(x[:len(x) // 2])
        odict['Opened'] = to_raw(x[len(x) // 2:])
    elif name == 'Rotate':
        odict['Right'] = to_raw(x[:len(x) // 2])
        odict['Left'] = to_raw(x[len(x) // 2:])
    else:
        odict[name] = to_raw(x)
    #elif 'FB' in name:
    #    odict['FB'] = to_raw(x)
    return odict

settings_file = 'D:\\vnd_spbu\\mock\\vnd_spbu_5days.json'
settings_file = 'D:\\vnd_spbu\\pilot\\mu5days\\vnd_spbu_5days.json'
#settings_file = 'C:\\Users\\nsmetanin\\Desktop\\results\\vnd_spbu_5days.json'
with open(settings_file, 'r', encoding="utf-8") as f:
    settings = loads(f.read())


from collections import OrderedDict
from pynfb.postprocessing.utils import load_rejections, get_info, fft_filter, dc_blocker, get_colors
cm = get_colors()


for j_subj, subj in enumerate(settings['subjects']):
    for j_exp, experiment in enumerate(subj):
        print(experiment)

        with h5py.File('{}\\{}\\{}'.format(settings['dir'], experiment, 'experiment_data.h5')) as f:
            fs, channels, p_names = get_info(f, settings['drop_channels'])
            info = mne.create_info(ch_names=channels, sfreq=fs, montage=mne.channels.read_montage(kind='standard_1005'),
                                   ch_types=['eeg' for ch in channels])
            raw_dict = OrderedDict()
            for j, name in enumerate(p_names[:3]):
                x = f['protocol{}/raw_data'.format(j + 1)][:]
                raw_dict = add_data_simple(raw_dict, name, x, info)


        f = plt.figure(figsize=(15, 10))
        for ax, idx in iter_topography(info, fig_facecolor='white', axis_facecolor='white', axis_spinecolor='k', fig=f):
            for key, raw in raw_dict.items():
                raw.plot_psd(dB=0, fmin=7, fmax=14, ax=ax, show=False, color=cm[key], picks=[idx],
                             n_fft=2048, n_overlap=1024)
                #print(np.mean(ax.get_xlim()))
                #ax.text(np.mean(ax.get_xlim()), np.mean(ax.get_ylim()), str(channels[idx]), color='r')
                ax.set_ylabel(str(channels[idx]))
                if channels[idx] == 'Fp2':
                    ax.legend(raw_dict.keys(), bbox_to_anchor=(5, 1.00))
                #ax.set_xticklabels([9, 10, 11])


        plt.gcf().suptitle('Power spectral densities')
        plt.gcf().savefig('S{}_D{}.png'.format(j_subj+1, j_exp+1), dpi=200)
        #plt.show()