from PyQt5 import QtGui, QtWidgets

import h5py
import pylab as plt
from scipy.signal import butter, lfilter, filtfilt
from scipy.linalg import eigh, inv, eig
import numpy as np

from pynfb.serializers.xml_ import get_lsl_info_from_xml
from pynfb.postprocessing.mu_experiment import dc_blocker, fft_filter
from pynfb.protocols.ssd.topomap_selector_ica import ICADialog


def csp3(x_dict, fs, band, butter_order=6, regularization_coef=0.1, lambda_=0.5):
    """
    """
    # apply filter
    cov_dict = {}
    for key, x in x_dict.items():
        x_filtered = fft_filter(x, fs, band)
        cov_dict[key] = np.dot(x_filtered.T, x_filtered) / x_filtered.shape[0]

    # find filters
    regularization = lambda z: z + regularization_coef * np.eye(z.shape[0])
    R1 = cov_dict['opened']
    R2 = (1-lambda_)*(cov_dict['closed'] - cov_dict['opened']) + lambda_*cov_dict['rotate']
    #print(R2)
    vals, vecs = eigh(regularization(R1), regularization(R2))
    vecs /= np.abs(vecs).max(0)

    # return vals, vecs and topographics (in descending order)
    reversed_slice = slice(-1, None, -1)
    topo = inv(vecs[:,reversed_slice]).T
    return vals[reversed_slice], vecs[:, reversed_slice], topo

if __name__ == '__main__':
    dir_ = 'D:\\vnd_spbu\\pilot\\mu5days'
    experiment = 'pilot5days_Rakhmankulov_Day3_03-01_12-51-41'
    with h5py.File('{}\\{}\\{}'.format(dir_, experiment, 'experiment_data.h5')) as f:
        ica = f['protocol1/signals_stats/left/rejections/rejection1'][:]

        x_filters = dc_blocker(np.dot(f['protocol1/raw_data'][:], ica))
        x_rotation = dc_blocker(np.dot(f['protocol2/raw_data'][:], ica))
        x_dict = {
            'closed': x_filters[:x_filters.shape[0] // 2],
            'opened': x_filters[x_filters.shape[0] // 2:],
            'rotate': x_rotation
        }
        drop_channels = ['AUX', 'A1', 'A2']
        labels, fs = get_lsl_info_from_xml(f['stream_info.xml'][0])
        print('fs: {}\nall labels {}: {}'.format(fs, len(labels), labels))
        channels = [label for label in labels if label not in drop_channels]



        scores, unmixing_matrix, topographies = csp3(x_dict, 250, (9, 14), lambda_=0.5, regularization_coef=0.01)
        app = QtWidgets.QApplication([])
        selector = ICADialog(np.vstack((x_filters, x_rotation)), channels, fs, unmixing_matrix=unmixing_matrix, mode='csp', scores=scores)
        selector.exec_()
