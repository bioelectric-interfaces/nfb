import mne
import os
import plotly_express as px
import numpy as np

from utils.load_results import load_data
import analysis_functions as af



# -----load the data from the H5 file
h5file = "/Users/2354158T/OneDrive - University of Glasgow/Documents/Pilot_2_RAW/PO5/0-nfb_task_PO2_05-10_11-39-12/experiment_data.h5"
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

# -----split the data into neurofeedback segments
# df1['protocol_change'] = df1['block_number'].diff()
# df1['p_change_events'] = df1.apply(lambda row: row.protocol_change if row.block_name == "nfb" else
# row.protocol_change * 2 if row.block_name == "start" else
# row.protocol_change * 3 if row.block_name == "cue" else
# row.protocol_change * 4 if row.block_name == "fc" else
# row.protocol_change * 5 if row.block_name == "end" else 0, axis=1)


# # Create the events list for the protocol transitions
# probe_events = df1[['p_change_events']].to_numpy()
# event_dict = {'nfb': 1, 'fc_w': 2, 'fc_b': 3, 'delay': 4, 'Input': 5}
#
# # Create the stim channel
# info = mne.create_info(['STI'], m_raw.info['sfreq'], ['stim'])
# stim_raw = mne.io.RawArray(probe_events.T, info)
# m_raw.add_channels([stim_raw], force_update_info=True)

# -----find the medians of the neurofeedback segments (raw AAI)
# Do this for the first 10000 samples
drop_cols = [x for x in df1.columns if x not in channels]
drop_cols.extend(['MKIDX', 'EOG', 'ECG', 'signal_AAI'])
eeg_data = df1.drop(columns=drop_cols)
eeg_data = eeg_data * 1e-6

aai_duration_samps = 10000
alpha_band=(8, 12)
chunksize = 20

mean_raw_l, std1_raw_l, pwr_raw_l = af.get_nfblab_power_stats_pandas(eeg_data[0:aai_duration_samps], fband=alpha_band,
                                                                     fs=fs,
                                                                     channel_labels=eeg_data.columns, chs=["PO7=1"],
                                                                     fft_samps=fs, chunksize=chunksize)

mean_raw_r, std1_raw_r, pwr_raw_r = af.get_nfblab_power_stats_pandas(eeg_data[0:aai_duration_samps], fband=alpha_band,
                                                                     fs=fs,
                                                                     channel_labels=eeg_data.columns, chs=["PO8=1"],
                                                                     fft_samps=fs, chunksize=chunksize)
aai_raw_left = (pwr_raw_l - pwr_raw_r) / (pwr_raw_l + pwr_raw_r)

aai_raw_left_median = np.median(aai_raw_left)

# -----convert the shortened data to mne format
df1_cropped = eeg_data[0:aai_duration_samps]
# create an MNE info
m_info = mne.create_info(ch_names=list(df1_cropped.columns), sfreq=fs,
                         ch_types=['eeg' for ch in list(df1_cropped.columns)])
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
m_raw = mne.io.RawArray(df1_cropped.T, m_info, first_samp=0, copy='auto', verbose=None)

# -----find the pwelch for the relevant channels
aai_pwelch = []
for t in np.arange(0, 10, 1):
    pwelch_l = mne.time_frequency.psd_welch(m_raw, fmin=8, fmax=12, n_fft=1000, picks=['PO7'], average='median', tmin=t, tmax=t+1, n_per_seg=20)
    pwelch_r = mne.time_frequency.psd_welch(m_raw, fmin=8, fmax=12, n_fft=1000, picks=['PO8'], average='median', tmin=t, tmax=t+1, n_per_seg=20)

    # calculate the AAI from this
    left_pwr = pwelch_l[0].mean()
    right_pwr = pwelch_r[0].mean()
    pwelch_aai = (left_pwr - right_pwr)/ (left_pwr + right_pwr)
    aai_pwelch.append(pwelch_aai)

alpha_raw = m_raw.filter(8, 12)

print("Done")