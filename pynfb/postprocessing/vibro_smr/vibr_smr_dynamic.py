import h5py
import numpy as np
from scipy import signal
import pandas as pd
import pylab as plt
from pynfb.signal_processing.helpers import get_outliers_mask
from pynfb.postprocessing.utils import get_info, fft_filter
from pynfb.protocols import SelectSSDFilterWidget
from pynfb.protocols.ssd.topomap_selector_ica import ICADialog
from PyQt5.QtWidgets import QApplication




# load data
def load_data(file_path):
    with h5py.File(file_path) as f:
        fs, channels, p_names = get_info(f, ['A1', 'A2'])
        data = [f['protocol{}/raw_data'.format(k + 1)][:] for k in range(len(p_names))]

        df = pd.DataFrame(np.concatenate(data), columns=channels)
        df['block_name'] = np.concatenate([[p]*len(d) for p, d in zip(p_names, data)])
        df['block_number'] = np.concatenate([[j + 1]*len(d) for j, d in enumerate(data)])

    return df, fs, p_names, channels
dir = r'C:\Users\Nikolai\Desktop\vibro-decay'
experiment = 'vibro-decay-vb-exp_01-26_17-42-57'
file = r'{}\{}\experiment_data.h5'.format(dir, experiment)


data, fs, p_names, channels = load_data(file)
for ch in []:
    channels.remove(ch)
    del data[ch]
data = data[~get_outliers_mask(data[channels], iter_numb=25, std=2.5)]
plt.plot(data[channels]*10000 + np.arange(len(channels)))
plt.show()

#data = data.iloc[~get_outliers_mask(data[channels], std=3, iter_numb=40)]
#data = data.drop(np.array(channels)[[0, 4, 9, 20]], axis=1)
#channels = [ch for j, ch in enumerate(channels) if j not in [0, 4, 9, 20]]
#data[channels] = fft_filter(data[channels], fs, (1, 45))
#data = data.iloc[~get_outliers_mask(data[channels], std=3, iter_numb=2)]
#plt.plot(data[channels] + 1000*np.arange(len(channels)))
#plt.show()


print('ww', sum(get_outliers_mask(data[channels][data['block_number']<3])))
# spatial filter (SMR detection)
try:
    filt = np.load('ica_{}.npy'.format(experiment))
    #filt = np.zeros(len(channels))
    #filt[channels.index('C3')] = 1
except FileNotFoundError:
    a = QApplication([])
    (rej, filt, topo, _unmix, _bandpass, _) = ICADialog.get_rejection(
        data[channels][data['block_number'].isin([1, 2])]#.iloc[~get_outliers_mask(data[channels][data['block_number']<12], std=2)]
        , channels, fs, mode='csp')
    #(_rej, filt, topo, _unmix, _bandpass, _) = ICADialog.get_rejection(np.concatenate(list(x[y=='Left']) + l1ist(x[y=='Legs'])), channels, fs, mode='csp')
    # filt, topography, bandpass, rejections = SelectSSDFilterWidget.select_filter_and_bandpass(np.concatenate(x), ch_names_to_2d_pos(channels), channels, sampling_freq=fs)
    np.save('ica_{}.npy'.format(experiment), filt)
data['SMR'] = np.dot(data[channels], filt)
#data = data.iloc[~get_outliers_mask(data[['C3', 'SMR']], std=3, iter_numb=10)]

# temporal filter
data['SMR_band_filt'] = fft_filter(data['SMR'], fs, (11, 13))
data['SMR_env'] = np.abs(signal.hilbert(data['SMR_band_filt']))

data.to_pickle('data-{}.pkl'.format(experiment))

data['SMR'].plot()
data['SMR_band_filt'].plot()
data['SMR_env'].plot()
plt.show()
