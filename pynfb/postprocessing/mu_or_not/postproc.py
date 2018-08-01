from scipy.io import loadmat
import pandas as pd
from scipy.signal import hilbert, welch
import pylab as plt
from pynfb.postprocessing.utils import fft_filter
import numpy as np
from pynfb.protocols.ssd.topomap_selector_ica import ICADialog

df = pd.read_pickle('p4')
fs = 1000
band = (8, 14)
df['SMRenv'] = pd.Series(fft_filter(df['SMR'], fs, band)).rolling(fs*100, center=True).std()
df['C3env'] = pd.Series(fft_filter(df['C3'], fs, band)).rolling(fs*100, center=True).std()

f, ax = plt.subplots(2)

ax[0].plot(df['C3'])
ax[0].plot(-df['SMR'])
ax[0].legend(['C3', 'SMR-Right-hand'])


ax[1].plot(df['C3env'])
ax[1].plot(df['SMRenv'])

ax[1].legend(['C3', 'SMR-Right-hand'])
#plt.plot(fft_filter(df['SMR'], fs, band), alpha=0.5)
plt.show()
