import mne
import pandas as pd
from utils.load_results import load_data
from mne.preprocessing import ICA
import plotly_express as px

# Ksenia
# h5file = r'C:\Users\2354158T\OneDrive - University of Glasgow\Documents\ksenia_20220809\0-nfb_task_ksenia_test_08-09_14-07-09\experiment_data.h5'

# Bethel
# h5file = r'C:\Users\2354158T\OneDrive - University of Glasgow\Documents\bethel_20220812\0-nfb_task_bethel_20220812_08-12_17-27-13\experiment_data.h5'
# bvfile = r'C:\Users\2354158T\OneDrive - University of Glasgow\Documents\bethel_20220812\bethel_20220812\bethel_20220812_tacs_test.vhdr'

# Rose
# h5file = r'C:\Users\2354158T\OneDrive - University of Glasgow\Documents\rose_test30092022\0-nfb_task_dry_run_20220928_09-30_12-21-58\experiment_data.h5'
h5file = r'C:\Users\2354158T\Documents\EEG_Data\rose_test30092022\0-baseline_rose_30092022_09-30_12-38-44\experiment_data.h5' # tACS switched off halfway through
h5file = r'C:\Users\2354158T\OneDrive - University of Glasgow\Documents\rose_test30092022\0-baseline_rose_30092022_09-30_11-48-43\experiment_data.h5'

def get_saturation_stats(row):
    length = len(row[row > 0.0001])
    return pd.Series([length,1,2])

df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

# Drop non eeg data
drop_cols = [x for x in df1.columns if x not in channels]
drop_cols.extend(['MKIDX', 'EOG'])
eeg_data = df1.drop(columns=drop_cols)

# Rescale the data (units are microvolts - i.e. x10^-6
eeg_data = eeg_data * 1e-6

# create an MNE info
m_info = mne.create_info(ch_names=list(eeg_data.columns), sfreq=fs,
                         ch_types=['eeg' if ch!= 'ECG' else 'eog' for ch in list(eeg_data.columns)])
# Set the montage (THIS IS FROM roi_spatial_filter.py)
standard_montage = mne.channels.make_standard_montage(kind='standard_1020')
standard_montage_names = [name.upper() for name in standard_montage.ch_names]
for j, channel in enumerate(eeg_data.columns):
    try:
        # make montage names uppercase to match own data
        standard_montage.ch_names[standard_montage_names.index(channel.upper())] = channel.upper()
    except ValueError as e:
        print(f"ERROR ENCOUNTERED: {e}")
m_info.set_montage(standard_montage, on_missing='ignore')

# Create the mne raw object with eeg data
m_raw = mne.io.RawArray(eeg_data.T, m_info, first_samp=0, copy='auto', verbose=None)
m_raw.plot()

lows = m_raw.copy().filter(l_freq=1., h_freq=40)
alpha = m_raw.copy().filter(l_freq=8, h_freq=12)

lows.plot()
tacs_nfb_raw = m_raw.copy()
tacs_nfb_raw.crop(tmin=18)
tacs_nfb_raw.plot()
tacs_nfb_epochs = mne.make_fixed_length_epochs(tacs_nfb_raw, duration=26, preload=True)
tacs_nfb_epochs.plot(picks=['P5'])


# SPECIFICALLY LOOKING AT CASE WHEN DEVICE WAS TURNED OFF - ROSE BASELINE
# ========================================================================
coupled_data = m_raw.copy().crop(tmin=0, tmax=17.5)
non_coupled_data = m_raw.copy().crop(tmin=45)
saturated_data = m_raw.copy().crop(tmin=18, tmax=44)

coupled_data.info['bads'] = ['O1', 'CZ', 'TP9', 'C2', 'PO7', 'FPZ', 'C1', 'P1', 'CP2']
non_coupled_data.info['bads'] = ['O1', 'CZ', 'TP9', 'C2', 'C1', 'PO7', 'FPZ', 'AFZ']
m_raw.info['bads'] = ['O1', 'CZ', 'TP9', 'C2', 'C1', 'PO7', 'FPZ', 'AFZ']

# ICA
ica_data = non_coupled_data.copy().filter(l_freq=1., h_freq=40)
ica = ICA(n_components=15, max_iter='auto', random_state=97)
ica.fit(non_coupled_data)
# m_high.drop_channels(['EOG', 'ECG'])
# Visualise
ica_data.load_data()
# ica.plot_sources(m_high, show_scrollbars=False)
ica.plot_components()


# Find the max, min, and mean length of zeros in the data
# =======================================================
# low_data = pd.DataFrame(lows.get_data()).T
# low_data = low_data.set_axis(list(eeg_data.columns), axis=1, inplace=False)
# low_data.FP1.plot()
eeg_sat_stats = eeg_data.apply(get_saturation_stats)


# look at the bv file from bethel
tacs_test = mne.io.read_raw_brainvision(bvfile, preload=True)
tacs_test_lows = tacs_test.copy().filter(l_freq=1., h_freq=40)

print("END")

