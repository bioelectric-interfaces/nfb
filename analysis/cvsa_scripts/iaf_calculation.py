"""
script to calculate a participant's alpha frequency from baseline
"""
from philistine.mne import savgol_iaf
from utils.load_results import load_data
import mne
import pandas as pd

h5file = "/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/0-baseline_ct_noise_test_04-27_13-27-54/experiment_data.h5"

h5file = "/Users/2354158T/OneDrive - University of Glasgow/Documents/cvsa_pilot_testing/lab_test_20220428/0-baseline_ct_test_04-28_16-49-26/experiment_data.h5"

# Put data in pandas data frame
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

# get baseline blocks and block numbers
df1_all_bl = df1.loc[df1['block_name'].str.contains("baseline", case=False)]
df1_all_bl = df1_all_bl.loc[~df1_all_bl['block_name'].str.contains("start", case=False)]
baseline_blocks = df1_all_bl['block_number'].unique()

iaf_data = []

for block in baseline_blocks:
    df1_bl = df1_all_bl.loc[df1_all_bl["block_number"] == block]
    baseline_name = df1_bl["block_name"].unique()[0]

    # Drop non eeg data
    eeg_data = df1_bl.drop(
        columns=['signal_left', 'signal_right', 'signal_AAI', 'events', 'reward', 'choice', 'answer', 'probe',
                 'block_name', 'chunk_n', 'cue', 'posner_stim', 'posner_time', 'response_data',
                 'block_number', 'sample'])
    eeg_data = eeg_data[['PO7', 'PO8']]

    # create an MNE info
    m_info = mne.create_info(ch_names=list(eeg_data.columns), sfreq=fs, ch_types=['eeg' for ch in list(eeg_data.columns)])

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

    # ----------get individual peak alpha
    iaf_raw = savgol_iaf(m_raw, pink_max_r2=1, fmax=14).PeakAlphaFrequency
    iaf_data.append(pd.DataFrame(
        dict(baseline_name=[baseline_name], block_number=[block],
             iaf=[iaf_raw])))
    print("----------------------------------------------")
    print(f"BASELINE: {baseline_name}, IAF: {iaf_raw}")
    print("----------------------------------------------")
print(iaf_data)