from pynfb.protocols.ssd.topomap_selector_ica import ICADialog
import mne
import pylab as plt
import pandas as pd
import numpy as np

def remove_bad_samples(data, window_size, threshold):
    df = pd.DataFrame(data)
    th = np.abs(df).rolling(window_size, center=True).max().max(1)
    good_samples_mask = th < threshold
    print('********Remove {} bad samples'.format(sum(~good_samples_mask)))
    return df.loc[good_samples_mask].values, good_samples_mask


N_EEG_CHANNELS = 30

raw = mne.io.read_raw_fif('/media/kolai/prispasobo/Kondrashin_raw.fif')
channels = raw.info['ch_names'][:N_EEG_CHANNELS]
fs = int(raw.info['sfreq'])



data = raw.load_data().notch_filter([50, 100]).filter(0.5, 40.).get_data()[:N_EEG_CHANNELS].T



# plt.plot(data[:fs*100])

ica_data, good_samples_mask = remove_bad_samples(data[:fs*60*10], window_size=fs*5, threshold=2000)

ica = ICADialog(ica_data, channels, fs)
ica.exec_()
# ica.table.get_checked_rows()

plt.plot(data[:, :]+np.arange(30)*2000, 'r')
plt.plot(ica.rejection.apply(data)[:, :]+np.arange(30)*2000, 'k')
plt.fill_between(np.arange(data.shape[0]), (~good_samples_mask).astype(int)*30*2000, color='r', alpha=0.4)
plt.gca().set_yticks(np.arange(30)*2000)
plt.gca().set_yticklabels(channels)
plt.xlim(0, fs*30)