"""
Script to clean data for analysis of pilot cvsa tests
"""
import mne
import plotly_express as px
import plotly.graph_objs as go

from analysis.cvsa_scripts.cvsa_analysis_2 import get_cue_dir, get_posner_time, read_log_file
from analysis.cvsa_scripts.eye_calibration import butter_lowpass_filter
from pynfb.protocols.ssd.csp import butter_bandpass_filter
from utils.load_results import load_data

# data paths

h5file = "/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-posner_task_test_psychopy_06-14_16-55-03/experiment_data.h5" # POSNER

# h5file = "/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-nfb_task_test_psychopy_06-14_17-13-14/experiment_data.h5" # Correct direction
# score = read_log_file("/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-nfb_task_test_psychopy_06-14_17-13-14/06-14_17-13-14.log")

# h5file = "/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-nfb_task_test_psychopy_06-14_17-41-40/experiment_data.h5" # wrong direction - left is normally lower here (looks like working)
# score = read_log_file("/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-nfb_task_test_psychopy_06-14_17-41-40/06-14_17-41-40.log")

# h5file = "/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-nfb_task_test_psychopy_06-14_17-27-31/experiment_data.h5" # pattern
# score = read_log_file("/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-nfb_task_test_psychopy_06-14_17-27-31/06-14_17-27-31.log")
#
# h5file = "/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-nfb_task_test_psychopy_06-14_17-53-44/experiment_data.h5" # nothing
# score = read_log_file("/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-nfb_task_test_psychopy_06-14_17-53-44/06-14_17-53-44.log")

# h5file = "/Users/christopherturner/Documents/EEG_Data/pilot2_COPY/PO5/0-nfb_task_PO2_05-10_11-18-09/experiment_data.h5"

# Load the h5 file
df1, fs, channels, p_names = load_data(h5file)
df1 = get_cue_dir(df1, channels=channels)
df1 = get_posner_time(df1)
df1['sample'] = df1.index

# Plot the online power channels
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df1['sample'], y=df1['signal_left_eye'],
                    mode='lines',
                    name='left_eye'))
fig1.add_trace(go.Scatter(x=df1['sample'], y=df1['signal_right_eye'],
                    mode='lines',
                    name='right_eye'))
fig1.add_trace(go.Scatter(x=df1['sample'], y=df1['signal_right'],
                    mode='lines',
                    name='right'))
fig1.add_trace(go.Scatter(x=df1['sample'], y=df1['signal_left'],
                    mode='lines',
                    name='left'))
fig1.show()

# Get the AAI for the left, right, and centre trials from signal_AAI----------------------------
cue_dirs = [1, 2, 3]
cue_recode = ["left", "right", "centre"]
df1['cue_dir'] = df1['cue_dir'].replace(cue_dirs, cue_recode)

# Get AAI by calculating raw signals from hdf5 (i.e. no smoothing)------------------------------------
# Drop non eeg data
# eeg_df = df1[df1['block_name'].str.contains("nfb")]
drop_cols = [x for x in df1.columns if x not in channels]
drop_cols.extend(['MKIDX', 'EOG', 'ECG', 'signal_AAI'])
eeg_data = df1.drop(columns=drop_cols)
eeg_data = eeg_data * 1e-6

# Convert to MNE data
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

# Drop all channels except PO7 and PO8
m_raw.drop_channels([x for x in list(eeg_data.columns) if x not in ['PO7', 'PO8']])

# set the reference to average
# m_raw = m_raw.set_eeg_reference(projection=False) # Be careful about using a projection vs actual data referencing in the analysis

# Check PSD of raw
m_raw.plot_psd()

# Filter alpha - check for noise
m_alpha = m_raw.copy()
alpha_range = (8.75, 10.75)
m_alpha.filter(l_freq=alpha_range[0], h_freq=alpha_range[1])

# Check PSD of filtered

# Look at EOG and ECG channels
lowcut = 5
highcut = 15
ecg_lp = butter_bandpass_filter(df1['EOG'], lowcut, highcut, fs, order=3, axis=0)
px.line(ecg_lp).show()
po7_lp = butter_bandpass_filter(df1['PO7'], lowcut, highcut, fs, order=3, axis=0)
px.line(po7_lp).show()

print("DONE")
