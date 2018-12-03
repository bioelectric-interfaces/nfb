from PyQt5 import QtGui, QtWidgets

from pynfb.serializers.xml_ import get_lsl_info_from_xml
from pynfb.signals import DerivedSignal
from pynfb.widgets.update_signals_dialog import SignalsSSDManager

from pynfb.widgets.helpers import ch_names_to_2d_pos
import numpy as np
import pickle
import os.path
import sys
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


experiments1 = [#'pilot_Nikolay_1_10-17_13-57-56',
                'pilot_Plackhin_1_10-20_12-03-01',
                'pilot_Tatiana_1_10-17_15-04-39',
                'pilot_Polyakova_1_10-24_15-21-18']

experiments2 = ['pilot_Nikolay_2_10-18_14-57-23',
                'pilot_Tatiana_2_10-18_16-00-44']

experiments = experiments1 + experiments2


import h5py



new_rejections_file = 'new_rejections.pkl'
if not os.path.isfile(new_rejections_file):

    a = QtWidgets.QApplication([])
    new_rejections = {}

    for experiment in experiments[:]:
        with h5py.File('{}\\{}\\{}'.format(pilot_dir, experiment, 'experiment_data.h5')) as f:
            print(experiment)
            old_rejections = [f['protocol1/signals_stats/left/rejections/rejection{}_topographies'.format(j + 1)][:] for j in range(2)]
            print('Old rejections ranks:', [old_rejection.shape[1] for old_rejection in old_rejections])

            raw = f['protocol{}/raw_data'.format(1)][:]

            labels_, fs_ = get_lsl_info_from_xml(f['stream_info.xml'][0])
            print(labels_)
            channels = [label for label in labels_ if label not in ['A1', 'A2', 'AUX']]
            print(labels_)

            pz_index = channels.index('Pz')
            raw = raw - np.dot(raw[:, [pz_index]], np.ones((1, raw.shape[1])))

            del channels[pz_index]
            raw = raw[:, np.arange(raw.shape[1]) != pz_index]


            signal = DerivedSignal(ind=0, name='Signal', bandpass_low=9, bandpass_high=14,
                                     spatial_filter=np.array([0]), n_channels=raw.shape[1])
            w = SignalsSSDManager([signal], raw, ch_names_to_2d_pos(channels), channels, None, None, [], sampling_freq=fs_ )
            w.exec_()

            rejections = signal.rejections.get_list()
            new_rejections[experiment] = rejections
    with open(new_rejections_file, 'wb') as pkl:
        pickle.dump(new_rejections, pkl)
    del a
else:
    print('file exist')
    with open(new_rejections_file, 'rb') as handle:
        new_rejections = pickle.load(handle)

print(new_rejections)

