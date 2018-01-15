from pynfb.postprocessing.utils import get_info, fft_filter
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt
from pynfb.signal_processing.decompositions import ICADecomposition
from mne.viz import plot_topomap
from pynfb.inlets.montage import Montage
from scipy.signal import hilbert

df = pd.read_csv('wow_ex.csv')
plt.plot(df['RightSMR-env'].rolling(500).mean())

df['RightSMR-env'] = np.log(df['RightSMR-env'].rolling(500).mean())
df = df.dropna()

sns.tsplot(df, time='times', unit='block_number', value='RightSMR-env', condition='block_name', err_style="unit_traces")
plt.show()