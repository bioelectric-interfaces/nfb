from scipy.io import loadmat
import pandas as pd
from scipy.signal import hilbert, welch
import pylab as plt
from pynfb.postprocessing.utils import fft_filter
from pynfb.signal_processing.helpers import get_outliers_mask
import numpy as np
from pynfb.protocols.ssd.topomap_selector_ica import ICADialog
from PyQt5 import QtGui, QtWidgets

fs = 1000
band = (8, 14)
channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9', 'Fc5', 'Fc1', 'Fc2', 'Fc6', 'Ft10', 'T3', 'C3', 'Cz',
            'C4', 'T4', 'Tp9', 'Cp5', 'Cp1', 'Cp2', 'Cp6', 'Tp10', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']
data = [loadmat(r'C:\Users\Nikolai\Desktop\Liza_diplom_data\Liza_diplom_data\treatment\p25\day2\proba{}.mat'.format(k))['X1Topo'].T for k in range(1, 16)]
df = pd.DataFrame(data=np.concatenate(data), columns=channels)

for ch in['Pz', 'Cz']:
    channels.remove(ch)
    del df[ch]
df = df.loc[~get_outliers_mask(df[channels])]
#plt.plot(*welch(df['C3'], fs, nperseg=4*fs))
plt.plot(df[channels])
plt.show()
#plt.show()
#df['C3env'] = np.abs(hilbert(fft_filter(df['C3'], fs, band)))
#df['C4env'] = np.abs(hilbert(fft_filter(df['C4'], fs, band)))
#plt.plot(df['C3env']+df['C4env'])

a = QtWidgets.QApplication([])
(rej, filt, topo, _unmix, _bandpass, _) = ICADialog.get_rejection(df.iloc[:fs*60*3], channels, fs)
df['SMR'] = np.dot(df.as_matrix(), filt)
df.to_pickle('p4')
plt.plot(df['SMR'])
plt.show()
