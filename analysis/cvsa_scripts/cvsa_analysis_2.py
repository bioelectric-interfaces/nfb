"""
Scropt to analyse covert attention protocols
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import platform
from scipy.stats import norm

from pynfb.signal_processing.filters import ExponentialSmoother, FFTBandEnvelopeDetector
from utils.load_results import load_data
import pandas as pd
import plotly_express as px
import plotly.graph_objs as go
import analysis.analysis_functions as af
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw, apply_inverse, apply_inverse_epochs

from pynfb.helpers import roi_spatial_filter as rsf
from philistine.mne import savgol_iaf

if platform.system() == "Windows":
    userdir = "2354158T"
else:
    userdir = "christopherturner"

task_data = {}
# h5file = f"/Users/{userdir}/Documents/EEG_Data/cvsa_test1/0-nfb_task_cvsa_test_04-06_17-35-45/experiment_data.h5"
# h5file = f"/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/0-test_task_cvsa_test_04-16_17-00-25/experiment_data.h5"
# h5file = f"/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/0-nfb_task_cvsa_test_04-22_16-09-15/experiment_data.h5"
# h5file = f"/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/0-test_task_ct_noise_test_04-26_10-00-42/experiment_data.h5"
# h5file = f"/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/0-test_task_ct_test_04-29_17-42-07/experiment_data.h5"

#--- lab test 28/04/22
h5file = "/Users/christopherturner/Documents/EEG_Data/cvsa_pilot_testing/lab_test_20220428/0-nfb_task_ct_test_04-28_17-39-22/experiment_data.h5"

# Put data in pandas data frame
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

channels.append("signal_AAI")

# Get the cued NFB side (preceding cue direction)
cue_dir_df = df1[(df1['cue'].isin([1,2,3])) & (df1['block_name'] == 'cue')]
cue_dir_key = {}
for index, row in cue_dir_df.iterrows():
    cue_dir_key[row['block_number']+1] = row['cue']
df1["cue_dir"] = df1['block_number'].map(cue_dir_key)


# Get the reaction times for valid and invalid trials
# Valid cue => Posner_stim == cue
# Invalid cue => posner_stim != cue
valid_cue_key = {}
for index, row in df1[df1['posner_stim'].isin([1,2])].iterrows():
    if row['posner_stim'] == row['cue_dir']:
        valid_cue_key[row['block_number']] = True
    elif row['cue_dir'] == 3:
        valid_cue_key[row['block_number']] = 'NA'
    else:
        valid_cue_key[row['block_number']] = False
df1["valid_cue"] = df1['block_number'].map(valid_cue_key)

# RT = time between 'posner_stim' and end of block
# Posner stim onset time = first time that isn't 0 in the block
grouped_df = df1[df1['posner_time'] >0].groupby("block_number", as_index=False)
stim_times = grouped_df.min()
reaction_time_key = {}
for index, row in stim_times.iterrows():
    block = df1[df1['block_number'] == row['block_number']]
    if not bool(block[block['response_data']>0].empty):
        response_time = block[block['response_data']>0].iloc[0]['response_data']
        reaction_time_key[row['block_number']] = response_time - row['posner_time']
df1["reaction_time"] = df1['block_number'].map(reaction_time_key)

# Plot reaction times for valid, invalid, and no training trials
rt_df = df1[['signal_AAI', 'block_name', 'block_number', 'sample', "reaction_time", 'posner_time', "valid_cue", 'posner_stim', 'cue', 'cue_dir']]
rt_df = rt_df[rt_df['block_name'].str.contains("nfb")]
rt_df = rt_df[rt_df['posner_stim'] != 0]
rt_df = rt_df.drop_duplicates(['block_number'], keep='first')


fig = px.violin(rt_df, x="valid_cue", y="reaction_time", box=True, points='all')
fig.show()


# Get the AAI for the left, right, and centre trials from signal_AAI----------------------------
aai_df = df1[['PO8', 'PO7', 'signal_AAI', 'block_name', 'block_number', 'sample', 'cue_dir']]
aai_df = aai_df[aai_df['block_name'].str.contains("nfb")]
grouped_aai_df = aai_df.groupby("block_number", as_index=False)
median_aais = grouped_aai_df.median()
fig = px.violin(median_aais, x="cue_dir", y="signal_AAI", box=True, points='all', range_y=[-0.3, 0.2])
fig.show()
# Plot the median of the left and right alphas from the online AAI signal
side_data = pd.melt(median_aais, id_vars=['cue_dir'], value_vars=['PO7', 'PO8'], var_name='side', value_name='data')
fig = px.box(side_data, x="cue_dir", y="data", color='side', points='all')
fig.show()


# Get AAI by calculating raw signals from hdf5 (i.e. no smoothing)------------------------------------
# Drop non eeg data
drop_cols = [x for x in df1.columns if x not in channels]
drop_cols.extend(['MKIDX', 'EOG', 'ECG', 'signal_AAI'])
eeg_data = df1.drop(columns=drop_cols)

# Rescale the data (units are microvolts - i.e. x10^-6
eeg_data = eeg_data * 1e-6
aai_duration_samps = df1.shape[0]#10000
alpha_band = (7.25, 11.25)
mean_raw_l, std1_raw_l, pwr_raw_l = af.get_nfblab_power_stats_pandas(eeg_data[0:aai_duration_samps], fband=alpha_band, fs=1000,
                                                             channel_labels=eeg_data.columns, chs=["PO7=1"],
                                                             fft_samps=1000)

mean_raw_r, std1_raw_r, pwr_raw_r = af.get_nfblab_power_stats_pandas(eeg_data[0:aai_duration_samps], fband=alpha_band, fs=1000,
                                                             channel_labels=eeg_data.columns, chs=["PO8=1"],
                                                             fft_samps=1000)
aai_raw_left = (pwr_raw_l - pwr_raw_r) / (pwr_raw_l + pwr_raw_r)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=eeg_data.index, y=aai_raw_left,
                    mode='lines',
                    name='AAI_calc'))
fig1.add_trace(go.Scatter(x=eeg_data.index, y=df1['signal_AAI'],
                    mode='lines',
                    name='AAI_online'))
fig1.show()

# Plot the median of the left and right alphas from the online AAI signal
aai_df_raw = df1[['PO8', 'PO7', 'block_name', 'block_number', 'sample', 'cue_dir']]
aai_df_raw['raw_aai'] = aai_raw_left
aai_df_raw = aai_df_raw[aai_df_raw['block_name'].str.contains("nfb")]
grouped_aai_df_raw = aai_df_raw.groupby("block_number", as_index=False)
median_aais_raw = grouped_aai_df_raw.median()
fig = px.violin(median_aais_raw, x="cue_dir", y="raw_aai", box=True, points='all', range_y=[-0.3, 0.2], title='raw_ref')
fig.show()
# Plot the median of the left and right alphas from the online AAI signal
side_data_raw = pd.melt(median_aais_raw, id_vars=['cue_dir'], value_vars=['PO7', 'PO8'], var_name='side', value_name='data')
fig = px.box(side_data_raw, x="cue_dir", y="data", color='side', points='all', title='raw_ref')
fig.show()


# Get AAI by calculating signals from MNE-------------------------------
# (try with avg and no referencing (no ref should be the same))

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
# m_raw = m_raw.set_eeg_reference(projection=False) # Be careful about using a projection vs actual data referencing in the analysis

raw_ref_avg_df = pd.DataFrame(m_raw.get_data(stop=aai_duration_samps).T, columns=m_raw.info.ch_names)
mean_avg_l, std_avg_l, pwr_avg_l = af.get_nfblab_power_stats_pandas(raw_ref_avg_df, fband=alpha_band, fs=1000,
                                                             channel_labels=m_raw.info.ch_names, chs=["PO7=1"],
                                                             fft_samps=1000)
mean_avg_r, std_avg_r, pwr_avg_r = af.get_nfblab_power_stats_pandas(raw_ref_avg_df, fband=alpha_band, fs=1000,
                                                             channel_labels=m_raw.info.ch_names, chs=["PO8=1"],
                                                             fft_samps=1000)
aai_avg_left = (pwr_avg_l - pwr_avg_r) / (pwr_avg_l + pwr_avg_r)
fig1.add_trace(go.Scatter(x=eeg_data.index, y=aai_avg_left,
                    mode='lines',
                    name='AAI_avg'))
fig1.show()

# Plot the median of the left and right alphas from the online AAI signal
df1['raw_aai'] = aai_avg_left
aai_df_avg = df1[['PO8', 'PO7', 'block_name', 'block_number', 'sample', 'cue_dir', 'raw_aai']]
aai_df_avg = aai_df_avg[aai_df_avg['block_name'].str.contains("nfb")]
grouped_aai_df_avg = aai_df_avg.groupby("block_number", as_index=False)
median_aais_avg = grouped_aai_df_avg.median()
fig = px.violin(median_aais_avg, x="cue_dir", y="raw_aai", box=True, points='all', range_y=[-0.3, 0.2], title='avg_ref')
fig.show()
# Plot the median of the left and right alphas from the online AAI signal
side_data_avg = pd.melt(median_aais_avg, id_vars=['cue_dir'], value_vars=['PO7', 'PO8'], var_name='side', value_name='data')
fig = px.box(side_data_avg, x="cue_dir", y="data", color='side', points='all', title='avg_ref')
fig.show()


# TODO:
#   Just use the already calculated AAI power for the whole experiment, Add a 'type' column and have '<x>_nfb', cue, fc'
conditions = [
    (df1['cue_dir'] == 1) & (df1['block_name'] == "nfb"),
    (df1['cue_dir'] == 2) & (df1['block_name'] == "nfb"),
    (df1['cue_dir'] == 3) & (df1['block_name'] == "nfb"),
    (df1['block_name'] == "fc"),
    (df1['block_name'] == "cue")
    ]

# create a list of the values we want to assign for each condition
values = ['left_nfb', 'right_nfb', 'centre_nfb', 'fc', 'cue']

# create a new column and use np.select to assign values to it using our lists as arguments
df1['block_type'] = np.select(conditions, values)

# Plot the different conditions
block_type_df = df1[['PO7', 'PO8', 'block_name', 'block_number', 'sample', 'cue_dir', 'raw_aai', 'block_type']]
grouped_block_type_df = block_type_df.groupby("block_number", as_index=False)
median_block_type = grouped_block_type_df.median(numeric_only=False)
median_block_type['block_type'] = grouped_block_type_df.first()['block_type']
fig = px.violin(median_block_type, x="block_type", y="raw_aai", box=True, points='all', range_y=[-0.3, 0.2], title='avg_ref')
fig.show()
fig = px.box(median_block_type, x="block_type", y="raw_aai", points='all', range_y=[-0.3, 0.2], title='avg_ref')
fig.show()
# Plot the median of the left and right alphas from the online AAI signal
block_type_side_data = pd.melt(median_block_type, id_vars=['block_type'], value_vars=['PO7', 'PO8'], var_name='side', value_name='data')
fig = px.box(block_type_side_data, x="block_type", y="data", color='side', points='all', title='avg_ref')
fig.show()




# Do the epoching and Add baseline and cue AAIs to the plot-----------------
# TODO:
#   2. look at epochs for all sections
#   3. plot against cues and fixation crosses

# Create the stim channel
# --- Get the probe events and MNE raw objects
# Get start of blocks as different types of epochs (1=start, 2=right, 3=left, 4=centre)
df1['protocol_change'] = df1['block_number'].diff()
df1['choice_events'] = df1.apply(lambda row: row.protocol_change if row.block_name == "start" else
                                 row.protocol_change * 2 if row.cue_dir == 1 else
                                 row.protocol_change * 3 if row.cue_dir == 2 else
                                 row.protocol_change * 4 if row.cue_dir == 3 else 0, axis=1)

# Create the events list for the protocol transitions
probe_events = df1[['choice_events']].to_numpy()
left_probe= 2
right_probe = 3
centre_probe = 4
event_dict = {'right_probe': right_probe, 'left_probe': left_probe, 'centre_probe': centre_probe}
info = mne.create_info(['STI'], m_raw.info['sfreq'], ['stim'])
stim_raw = mne.io.RawArray(probe_events.T, info)
m_raw.add_channels([stim_raw], force_update_info=True)

# Get the epoch object
m_filt = m_raw.copy()
m_filt.filter(8, 14, n_jobs=1,  # use more jobs to speed up.
               l_trans_bandwidth=1,  # make sure filter params are the same
               h_trans_bandwidth=1)  # in each band and skip "auto" option.

events = mne.find_events(m_raw, stim_channel='STI')
reject_criteria = dict(eeg=100e-6)

left_chs = ['PO7=1']
right_chs = ['PO8=1']

epochs = mne.Epochs(m_filt, events, event_id=event_dict, tmin=-2, tmax=10, baseline=None,
                    preload=True, detrend=1)#, reject=reject_criteria)
# epochs.drop([19,22,27,32]) # Drop bads for K's 1st dataset

# fig = epochs.plot(events=events)

probe_left = epochs['left_probe'].average()
probe_right = epochs['right_probe'].average()

dataframes = []
# ----Look at the power for the epochs in the left and right channels for left and right probes
e_mean1, e_std1, epoch_pwr1 = af.get_nfb_epoch_power_stats(epochs['left_probe'], fband=alpha_band, fs=1000, channel_labels=epochs.info.ch_names, chs=["PO7=1"])
e_mean2, e_std2, epoch_pwr2 = af.get_nfb_epoch_power_stats(epochs['left_probe'], fband=alpha_band, fs=1000, channel_labels=epochs.info.ch_names,  chs=["PO8=1"])

df_i = pd.DataFrame(dict(left=e_mean1, right=e_mean2))
df_i['section'] = f"left_probe"
dataframes.append(df_i)

dataframes_aai_cue = []
dataframes_aai = []
aai_nfb_left = (epoch_pwr1 - epoch_pwr2) / (epoch_pwr1 + epoch_pwr2)
df_ix = pd.DataFrame(dict(aai=aai_nfb_left.mean(axis=0)[0]))
df_ix['probe'] = f"left"
dataframes_aai.append(df_ix[2000:9000])
dataframes_aai_cue.append(df_ix[0:2000])
fig2 = go.Figure()
af.plot_nfb_epoch_stats(fig2, aai_nfb_left.mean(axis=0)[0], aai_nfb_left.std(axis=0)[0], name=f"left probe aai",
                     title=f"left probe aai",
                     color=(230, 20, 20, 1), y_range=[-1, 1])


e_mean1, e_std1, epoch_pwr1 = af.get_nfb_epoch_power_stats(epochs['right_probe'], fband=alpha_band, fs=1000, channel_labels=epochs.info.ch_names, chs=["PO7=1"])
e_mean2, e_std2, epoch_pwr2 = af.get_nfb_epoch_power_stats(epochs['right_probe'], fband=alpha_band, fs=1000, channel_labels=epochs.info.ch_names,  chs=["PO8=1"])

df_i = pd.DataFrame(dict(left=e_mean1, right=e_mean2))
df_i['section'] = f"right_probe"
dataframes.append(df_i)

aai_nfb_right = (epoch_pwr1 - epoch_pwr2) / (epoch_pwr1 + epoch_pwr2)
df_ix = pd.DataFrame(dict(aai=aai_nfb_right.mean(axis=0)[0]))
df_ix['probe'] = f"right"
dataframes_aai.append(df_ix[2000:9000])
dataframes_aai_cue.append(df_ix[0:2000])
af.plot_nfb_epoch_stats(fig2, aai_nfb_right.mean(axis=0)[0], aai_nfb_right.std(axis=0)[0], name=f"right probe aai",
                     title=f"mean aai time course",
                     color=(20, 20, 230, 1), y_range=[-1, 1])


e_mean1, e_std1, epoch_pwr1 = af.get_nfb_epoch_power_stats(epochs['centre_probe'], fband=alpha_band, fs=1000, channel_labels=epochs.info.ch_names, chs=["PO7=1"])
e_mean2, e_std2, epoch_pwr2 = af.get_nfb_epoch_power_stats(epochs['centre_probe'], fband=alpha_band, fs=1000, channel_labels=epochs.info.ch_names,  chs=["PO8=1"])

df_i = pd.DataFrame(dict(left=e_mean1, right=e_mean2))
df_i['section'] = f"centre_probe"
dataframes.append(df_i)

aai_nfb_centre = (epoch_pwr1 - epoch_pwr2) / (epoch_pwr1 + epoch_pwr2)
df_ix = pd.DataFrame(dict(aai=aai_nfb_centre.mean(axis=0)[0]))
df_ix['probe'] = f"centre"
dataframes_aai.append(df_ix[2000:9000])
dataframes_aai_cue.append(df_ix[0:2000])
af.plot_nfb_epoch_stats(fig2, aai_nfb_centre.mean(axis=0)[0], aai_nfb_centre.std(axis=0)[0], name=f"centre probe aai",
                     title=f"mean aai time course",
                     color=(20, 230, 20, 1), y_range=[-1, 1])
fig2.add_vline(x=2000, annotation_text="cvsa start")
fig2.show()

fig2.show()







aai_section_df = pd.concat(dataframes_aai)
aai_section_df = aai_section_df.melt(id_vars=['probe'], var_name='side', value_name='data')
px.box(aai_section_df, x='probe', y='data', title="post aai task").show()


# Plot the AAI Boxes
figb = go.Figure()
# bl_aai_df1 = pd.DataFrame(dict(bl_eo1=aai_baseline_eo1.mean(axis=0)[0],
#                               bl_ec1=aai_baseline_ec1.mean(axis=0)[0]))
# bl_aai_df1 = bl_aai_df1.melt(var_name='baseline_type', value_name='data')
# figb.add_trace(go.Box(x=bl_aai_df1['baseline_type'], y=bl_aai_df1['data'],
#                       notched=True,
#                       line=dict(color='blue')))


# TODO - NOTE - NEED TO TAKE OUT BEGINNING OF EPOCHS HERE BECAUSE THEY SHOULDN"T COUNT TO THE MEANS ETC
figb.add_trace(go.Box(y=aai_section_df.loc[aai_section_df['probe'] == 'left']['data'],name='left task') )
figb.add_trace(go.Box(y=aai_section_df.loc[aai_section_df['probe'] == 'right']['data'],name='right task') )
figb.add_trace(go.Box(y=aai_section_df.loc[aai_section_df['probe'] == 'centre']['data'],name='centre task') )

figb.show()
aai_section_df_cue = pd.concat(dataframes_aai_cue)
aai_section_df_cue = aai_section_df_cue.melt(id_vars=['probe'], var_name='side', value_name='data')
px.box(aai_section_df_cue, x='probe', y='data', title="post aai cue").show()

figb.add_trace(go.Box(y=aai_section_df_cue.loc[aai_section_df_cue['probe'] == 'left']['data'],name='left cue') )
figb.add_trace(go.Box(y=aai_section_df_cue.loc[aai_section_df_cue['probe'] == 'right']['data'],name='right cue') )
figb.add_trace(go.Box(y=aai_section_df_cue.loc[aai_section_df_cue['probe'] == 'centre']['data'],name='centre cue') )
figb.update_layout(
    title="section aai",
    hovermode="x",
)
figb.show()






# look at the channels together
e_mean1, e_std1, epoch_pwr = af.get_nfb_epoch_power_stats(epochs['right_probe'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names, chs=["PO7=1"])
e_mean2, e_std2, epoch_pwr = af.get_nfb_epoch_power_stats(epochs['left_probe'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names,  chs=["PO7=1"])
fig = go.Figure()
af.plot_nfb_epoch_stats(fig, e_mean1, e_std1, name="right_probe", title="left chs", color=(230, 20, 20, 1), y_range=[-0.2e-6, 4e-6])
af.plot_nfb_epoch_stats(fig, e_mean2, e_std2, name="left_probe", title="left chs", color=(20, 20, 220, 1), y_range=[-0.2e-6, 4e-6])
fig.show()

e_mean1, e_std1, epoch_pwr = af.get_nfb_epoch_power_stats(epochs['right_probe'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names, chs=["PO8=1"])
e_mean2, e_std2, epoch_pwr = af.get_nfb_epoch_power_stats(epochs['left_probe'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names,  chs=["PO8=1"])
fig = go.Figure()
af.plot_nfb_epoch_stats(fig, e_mean1, e_std1, name="right_probe", title="right chs", color=(230, 20, 20, 1), y_range=[-0.2e-6, 4e-6])
af.plot_nfb_epoch_stats(fig, e_mean2, e_std2, name="left_probe", title="lefrightt chs", color=(20, 20, 220, 1), y_range=[-0.2e-6, 4e-6])
fig.show()




bl_aai_df2 = pd.DataFrame(dict(bl_eo2=aai_baseline_eo2.mean(axis=0)[0],
                              bl_ec2=aai_baseline_ec2.mean(axis=0)[0]))
bl_aai_df2 = bl_aai_df2.melt(var_name='baseline_type', value_name='data')
figb.add_trace(go.Box(x=bl_aai_df2['baseline_type'], y=bl_aai_df2['data'],
                      notched=True,
                      line=dict(color='blue')))
figb.show()



# -- LOOk at the left and right channels separately
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

