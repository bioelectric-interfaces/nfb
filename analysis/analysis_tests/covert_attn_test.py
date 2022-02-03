import mne

from utils.load_results import load_data
import pandas as pd
import plotly_express as px
import analysis.analysis_functions as af

task_data = {}
h5file = "/Users/christopherturner/Documents/EEG_Data/system_testing/attn_right_02-01_15-44-49/experiment_data.h5"

# Put data in pandas data frame
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

channels.append("signal_AAI_sc")
channels.append("signal_AAI_so")

df2 = pd.melt(df1, id_vars=['block_name', 'block_number', 'sample', 'choice', 'probe', 'answer', 'reward'],
                  value_vars=channels, var_name="channel", value_name='data')
aai_sc_data = df2.loc[df2['channel'] == "signal_AAI_sc"].reset_index(drop=True)
aai_so_data = df2.loc[df2['channel'] == "signal_AAI_so"].reset_index(drop=True)

left_electrode = df2.loc[df2['channel'] == "O2"].reset_index(drop=True)
right_electrode = df2.loc[df2['channel'] == "signal_AAI_so"].reset_index(drop=True)

fig = px.line(aai_sc_data, x=aai_sc_data.index, y="data", color='block_name', title=f"scalp aai")
fig.show()
fig = px.line(aai_so_data, x=aai_so_data.index, y="data", color='block_name', title=f"scalp aai")
fig.show()
fig = px.box(aai_sc_data, x='block_name', y="data", title="scalp aai")
fig.show()
fig = px.box(aai_so_data, x='block_name', y="data", title='source aai')
fig.show()
pass



# --- look at just certain electrodes
# Get start of blocks as different types of epochs (1=start, 2=right, 3=left, 4=centre)
df1['protocol_change'] = df1['block_number'].diff()
df1['choice_events'] = df1.apply(lambda row: row.protocol_change if row.block_name == "start" else
                                 row.protocol_change * 2 if row.block_name == "right" else
                                 row.protocol_change * 3 if row.block_name == "left" else
                                 row.protocol_change * 4 if row.block_name == "centre" else 0, axis=1)

# Create the events list for the protocol transitions
probe_events = df1[['choice_events']].to_numpy()
right_probe = 2
left_probe = 3
centre_probe = 4
event_dict = {'right_probe': right_probe, 'left_probe': left_probe, 'centre_probe': centre_probe}

# Drop non eeg data
drop_cols = [x for x in df1.columns if x not in channels]
drop_cols.extend(['MKIDX', 'EOG', 'ECG', "signal_AAI_sc", "signal_AAI_so", 'protocol_change', 'choice_events'])

eeg_data = df1.drop(columns=drop_cols)

# Rescale the data (units are microvolts - i.e. x10^-6
eeg_data = eeg_data * 1e-6
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

# set the reference to average
m_raw.set_eeg_reference(projection=True)

# Create the stim channel
info = mne.create_info(['STI'], m_raw.info['sfreq'], ['stim'])
stim_raw = mne.io.RawArray(probe_events.T, info)
m_raw.add_channels([stim_raw], force_update_info=True)

# Get alpha
m_filt = m_raw.copy()
m_filt.filter(8, 14, n_jobs=1,  # use more jobs to speed up.
               l_trans_bandwidth=1,  # make sure filter params are the same
               h_trans_bandwidth=1)  # in each band and skip "auto" option.

# Remove all channels except ones of interest
m_filt.drop_channels([ch for ch in m_info.ch_names if ch not in ['O1', 'O2', 'P3', 'P4', 'PO7', 'PO8', 'STI']])

events = mne.find_events(m_raw, stim_channel='STI')
epochs = mne.Epochs(m_filt, events, event_id=event_dict, tmin=-0.1, tmax=18,
                    preload=True, detrend=1)
fig = epochs.plot(events=events)

probe_left = epochs['left_probe'].average()
probe_right = epochs['right_probe'].average()
probe_centre = epochs['centre_probe'].average()
fig2 = probe_left.plot(spatial_colors=True)
fig2 = probe_right.plot(spatial_colors=True)

# plot topomap
probe_left.plot_topomap(times=[-0.1, 0.1, 0.4], average=0.05)
probe_right.plot_topomap(times=[-0.1, 0.1, 0.4], average=0.05)
probe_left.plot_joint(title="left")
probe_right.plot_joint(title="right")

# LOOK AT ENVELOPE
probe_left.apply_hilbert(picks=['O1', 'O2', 'P3', 'P4', 'PO7', 'PO8'], envelope=True, n_jobs=1, n_fft='auto', verbose=None)
probe_right.apply_hilbert(picks=['O1', 'O2', 'P3', 'P4', 'PO7', 'PO8'], envelope=True, n_jobs=1, n_fft='auto', verbose=None)
fig2 = probe_left.plot(spatial_colors=True)
fig2 = probe_right.plot(spatial_colors=True)

# Look at left and right evoked
left_chs = ['O1', 'P3', 'PO7']#['O1', 'PO3', 'PO7', 'P1', 'P3', 'P5', 'P7', 'P9', 'PZ', 'P0Z']
right_chs = ['O2', 'P4', 'PO8']#['O2', 'PO4', 'PO8', 'P2', 'P3', 'P6', 'P7', 'P10', 'PZ', 'P0Z']
picks = left_chs + right_chs
# picks = ['P7', 'PO7', 'O1', 'OZ', 'PO8', 'P8', 'PO3', 'POZ', 'PO4', 'PO8']
evokeds = dict(left_probe=list(epochs['left_probe'].iter_evoked()),
               right_probe=list(epochs['right_probe'].iter_evoked()))
mne.viz.plot_compare_evokeds(evokeds, combine='mean', picks=picks)

# Look at left side vs right side for left probe
left_ix = mne.pick_channels(probe_left.info['ch_names'], include=right_chs)
right_ix = mne.pick_channels(probe_left.info['ch_names'], include=left_chs)
roi_dict = dict(left_ROI=left_ix, right_ROI=right_ix)
roi_evoked = mne.channels.combine_channels(probe_left, roi_dict, method='mean')
print(roi_evoked.info['ch_names'])
roi_evoked.plot()


# # Get signal envelope of left and right chans
# m_env = m_raw.copy()
#
# # Get alpha
# m_env.filter(8, 14, n_jobs=1,  # use more jobs to speed up.
#                l_trans_bandwidth=1,  # make sure filter params are the same
#                h_trans_bandwidth=1)  # in each band and skip "auto" option.
#
# m_env.drop_channels([ch for ch in m_info.ch_names if ch not in ['O2', 'O1']])
# m_env.apply_hilbert(picks=['O2', 'O1'], envelope=True, n_jobs=1, n_fft='auto', verbose=None)
pass