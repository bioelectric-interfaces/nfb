

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import platform

from pynfb.signal_processing.filters import ExponentialSmoother, FFTBandEnvelopeDetector
from utils.load_results import load_data
import pandas as pd
import plotly_express as px
import plotly.graph_objs as go
import analysis.analysis_functions as af
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw, apply_inverse


if platform.system() == "Windows":
    userdir = "2354158T"
else:
    userdir = "christopherturner"

task_data = {}
h5file = f"/Users/{userdir}/Documents/EEG_Data/system_testing/bar_plot_attn/0-nfb_task_ct_bartest_02-14_19-22-01/experiment_data.h5" # Bar test (left, left, right, right) light on
# h5file = f"/Users/{userdir}/Documents/EEG_Data/system_testing/bar_plot_attn/0-nfb_task_ct_plottest_02-14_19-47-43/experiment_data.h5" # plot test (left, right, left, right) light off


# h5file = f"/Users/{userdir}/Documents/EEG_Data/pilot_202201/kk/source/1-nfb_task_kk_02-15_14-22-12/experiment_data.h5" #Ksenia bar neurofeedback1
# h5file = f"/Users/{userdir}/Documents/EEG_Data/pilot_202201/kk/source/1-nfb_task_kk_02-15_14-48-45/experiment_data.h5" #Ksenia bar neurofeedback2

# Put data in pandas data frame
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

# Drop noisy data !!THIS SEEMS TO CAUSE THE EPOCHS TO BUG OUT - DUPLICATES AND RESHUFFLES THE CHANNELS - MAYBE COS OF THE MONTAGE?
# keep_chs = ['PO7', 'PO8', 'EOG', 'ECG', 'signal_Alpha_Left', 'signal_Alpha_Right', 'signal_AAI', 'reward', 'block_name', 'block_number', 'sample']
# drop_cols = [x for x in df1.columns if x not in keep_chs]
# df1 = df1.drop(columns=drop_cols)

# --- Get the probe events and MNE raw objects
# Get start of blocks as different types of epochs (1=start, 2=right, 3=left, 4=centre)
df1['protocol_change'] = df1['block_number'].diff()
df1['nfb_events'] = df1.apply(lambda row: row.protocol_change if row.block_name == "NFB" else 0, axis=1)

# Create the events list for the protocol transitions
probe_events = df1[['nfb_events']].to_numpy()
event_dict = {'nfb': 1}

# Drop non eeg data (and noisy channels)

drop_cols = [x for x in df1.columns if x not in channels]
drop_cols.extend(['MKIDX', 'EOG', 'ECG', 'signal_Alpha_Left', 'signal_Alpha_Right', 'signal_AAI', 'reward', 'block_name', 'block_number', 'sample']) # NEED ALL CHS - something goes wrong if you just have 2 channels (one is a reflection of the other)
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


# Epoch data (combine all nfb protocols as epochs)

# Get the epoch object
m_filt = m_raw.copy()
m_filt.filter(8, 14, n_jobs=1,  # use more jobs to speed up.
               l_trans_bandwidth=1,  # make sure filter params are the same
               h_trans_bandwidth=1)  # in each band and skip "auto" option.

events = mne.find_events(m_raw, stim_channel='STI')
reject_criteria = dict(eeg=100)

# TODO: try marking bad channels before this step
epochs = mne.Epochs(m_filt, events, event_id=event_dict, tmin=0, tmax=10, baseline=None,
                    preload=True, detrend=1)#, reject=reject_criteria)

# fig = epochs.plot(events=events)

left_chs = ['PO7=1']
right_chs = ['PO8=1']
# left_chs = ["CP5=1", "P5=1", "O1=1"]
# right_chs = ["CP6=1", "P6=1", "O2=1"]


# DO BASELINE STUFF
fig_bl_eo1, aai_baseline_eo1, bl_dataframe_eo1 = af.do_baseline_epochs(df1, m_filt, left_chs, right_chs, fig=None, fb_type="active", baseline_name='bl_eo1', block_number=4)
fig_bl_ec1, aai_baseline_ec1, bl_dataframe_ec1 = af.do_baseline_epochs(df1, m_filt, left_chs, right_chs, fig=None, fb_type="active", baseline_name='bl_ec1', block_number=6)
fig_bl_eo2, aai_baseline_eo2, bl_dataframe_eo2 = af.do_baseline_epochs(df1, m_filt, left_chs, right_chs, fig=None, fb_type="active", baseline_name='bl_eo2', block_number=212)
fig_bl_ec2, aai_baseline_ec2, bl_dataframe_ec2 = af.do_baseline_epochs(df1, m_filt, left_chs, right_chs, fig=None, fb_type="active", baseline_name='bl_ec2', block_number=214)

fig_bl_aais = go.Figure()
af.plot_nfb_epoch_stats(fig_bl_aais, aai_baseline_eo1.mean(axis=0)[0], aai_baseline_eo1.std(axis=0)[0], name="bl_eo_1", title="baseline aais", color=(230, 20, 20, 1), y_range=[-0.7, 0.7])
af.plot_nfb_epoch_stats(fig_bl_aais, aai_baseline_ec1.mean(axis=0)[0], aai_baseline_ec1.std(axis=0)[0], name="bl_ec_1", title="baseline aais", color=(20, 220, 20, 1), y_range=[-0.7, 0.7])
af.plot_nfb_epoch_stats(fig_bl_aais, aai_baseline_eo2.mean(axis=0)[0], aai_baseline_eo2.std(axis=0)[0], name="bl_eo_2", title="baseline aais", color=(230, 20, 20, 1), y_range=[-0.7, 0.7])
af.plot_nfb_epoch_stats(fig_bl_aais, aai_baseline_ec2.mean(axis=0)[0], aai_baseline_ec2.std(axis=0)[0], name="bl_ec_2", title="baseline aais", color=(20, 220, 20, 1), y_range=[-0.7, 0.7])
fig_bl_aais.show()

# ----Look at the power for the epochs in the left and right channels for left and right probes
e_mean1, e_std1, epoch_pwr1 = af.get_nfb_epoch_power_stats(epochs['nfb'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names, chs=left_chs)
e_mean2, e_std2, epoch_pwr2 = af.get_nfb_epoch_power_stats(epochs['nfb'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names,  chs=right_chs)
fig = go.Figure()
af.plot_nfb_epoch_stats(fig, e_mean1, e_std1, name="left_chs", title="nfb", color=(230, 20, 20, 1), y_range=[-0.2e-6, 4e-6])
af.plot_nfb_epoch_stats(fig, e_mean2, e_std2, name="right_chs", title="nfb", color=(20, 220, 20, 1), y_range=[-0.2e-6, 4e-6])
fig.show()

# Look at epochs in time quarters
dataframes, dataframes_aai = af.do_quartered_epochs(epochs, left_chs, right_chs, fb_type="active")
dataframes_nfb = []
colors = ['blue', 'red']
for s in dataframes:
    dataframes_nfb.append(s[-5000:])
section_df_nfb = pd.concat(dataframes_nfb)
section_df_nfb = section_df_nfb.melt(id_vars=['section'], var_name='side', value_name='data')
section_df_bl_eo_1 = bl_dataframe_eo1.melt(id_vars=['section'], var_name='side', value_name='data')
section_df_bl_ec_1 = bl_dataframe_ec1.melt(id_vars=['section'], var_name='side', value_name='data')
section_df_bl_eo_2 = bl_dataframe_eo2.melt(id_vars=['section'], var_name='side', value_name='data')
section_df_bl_ec_2 = bl_dataframe_ec2.melt(id_vars=['section'], var_name='side', value_name='data')
section_df = pd.concat([section_df_bl_eo_1,section_df_bl_ec_1, section_df_nfb, section_df_bl_eo_2,section_df_bl_ec_2], ignore_index=True)
fig=go.Figure()
for i, side in enumerate(section_df['side'].unique()):
    df_plot = section_df[section_df['side'] == side]
    fig.add_trace(go.Box(x=df_plot['section'], y=df_plot['data'],
                         notched=True,
                         line=dict(color=colors[i]),
                         name='side=' + side))
# Append the baseline and plot
fig.update_layout(boxmode='group', xaxis_tickangle=1,yaxis_range=[0.7e-6,6.1e-6], title=f"KK bar NFB1 - {','.join(left_chs + right_chs)}")
fig.show()
# TODO: add baseline AAIs

# TODO:
#     [ ] look at online AAI and do box plots comparing to calculated
#     [ ] look at score and plot trend to see if it matches the AAI