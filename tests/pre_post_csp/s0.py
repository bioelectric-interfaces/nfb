from mne.io import read_epochs_eeglab
import numpy as np
import pylab as plt
from scipy.io import loadmat
from pynfb.widgets.helpers import ch_names_to_2d_pos
from pynfb.protocols.ssd.topomap_selector_ica import ICADialog
from PyQt5.QtWidgets import QApplication

fs = 258


pre = loadmat('C:\\Users\\nsmetanin\\Downloads\\nfb_bci\\wetransfer-07cfaf\\Subj01_POSTdata.mat')
channels = [a[0] for a in pre['EEGchanslabels'][0]]
pre = pre['EEGPOSTdata']

post = loadmat('C:\\Users\\nsmetanin\\Downloads\\nfb_bci\\wetransfer-07cfaf\\Subj01_PREdata.mat')['EEGPREdata']


print(pre.shape, post.shape, channels)

#print(ch_names_to_2d_pos(channels))
a = QApplication([])
n_samples = 50
print(pre[:, -n_samples:, 0].shape)
x = np.concatenate([np.concatenate([pre[:, -n_samples:, k].T for k in range(pre.shape[2])]),
                                        np.concatenate([post[:, :n_samples, k].T for k in range(post.shape[2])])])
print(x.shape)
ICADialog.get_rejection(x, channels, fs, mode='csp')
