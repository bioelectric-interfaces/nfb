# TODO:
#   look at the psd of the whole signal & just occipital electrodes to find the peak alpha
#   make sure you're actually including this in the filter
#   Run this through the NFB auto - freq detection to see if it detects it
#   Also run through with CSP to see if it detects left/right attn
#   look at nfb alpha lateralisation for each block with the different electrode setups


import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import platform

from mne.time_frequency import tfr_morlet

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

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne.decoding import CSP

if platform.system() == "Windows":
    userdir = "2354158T"
else:
    userdir = "christopherturner"

task_data = {}
# h5file = f"/Users/{userdir}/Documents/EEG_Data/system_testing/ksenia_cvsa/cvsa_02-05_15-39-15/experiment_data.h5" # Ksenia cvsa tasks 1
# h5file = f"/Users/{userdir}/Documents/EEG_Data/system_testing/ksenia_cvsa/cvsa_02-05_15-47-03/experiment_data.h5" # Ksenia cvsa tasks 2

# h5file = f"/Users/{userdir}/Documents/EEG_Data/system_testing/bar_plot_attn/ctcvsa_02-14_20-13-40/experiment_data.h5" # Chris cvsa 1 (with baselines)
#
h5file = f"/Users/{userdir}/Documents/EEG_Data/system_testing/ksenia_cvsa/cvsa_02-15_15-21-37/experiment_data.h5" # Ksenia cvsa 3 **
# h5file = f"/Users/{userdir}/Documents/EEG_Data/system_testing/ksenia_cvsa/cvsa_02-15_15-37-24/experiment_data.h5" # Ksenia cvsa 4

# Put data in pandas data frame
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

channels.append("signal_AAI")

df2 = pd.melt(df1, id_vars=['block_name', 'block_number', 'sample', 'choice', 'probe', 'answer', 'reward'],
                  value_vars=channels, var_name="channel", value_name='data')
aai_sc_data = df2.loc[df2['channel'] == "signal_AAI"].reset_index(drop=True)

left_electrode = df2.loc[df2['channel'] == "PO7"].reset_index(drop=True)
right_electrode = df2.loc[df2['channel'] == "PO8"].reset_index(drop=True)

fig = px.line(aai_sc_data, x=aai_sc_data.index, y="data", color='block_name', title=f"scalp aai")
fig.show()
fig = px.box(aai_sc_data, x='block_name', y="data", title="scalp aai")
fig.show()
pass

# ------- NFB LAB FILTERING
# Get left attention condition
eeg_data_pl = df1.loc[df1['block_name'] == 'probe_left']
drop_cols = [x for x in df1.columns if x not in channels]
drop_cols.extend(['MKIDX', 'EOG', 'ECG', 'signal_AAI'])
eeg_data_pl = eeg_data_pl.drop(columns=drop_cols)
# Get right attention condition
eeg_data_pr = df1.loc[df1['block_name'] == 'probe_right']
drop_cols = [x for x in df1.columns if x not in channels]
drop_cols.extend(['MKIDX', 'EOG', 'ECG', 'signal_AAI'])
eeg_data_pr = eeg_data_pr.drop(columns=drop_cols)

# Rescale the data (units are microvolts - i.e. x10^-6
eeg_data_pl = eeg_data_pl * 1e-6
eeg_data_pr = eeg_data_pr * 1e-6

bandpass = (8, 14) # TODO - get this right
smoothing_factor = 0.7
smoother = ExponentialSmoother(smoothing_factor)
n_samples = 1000
signal_estimator = FFTBandEnvelopeDetector(bandpass, fs, smoother, n_samples)

left_alpha_chs = "PO7=1"#;P5=1;O1=1"
right_alpha_chs = "PO8=1"
channel_labels = eeg_data_pl.columns

left_derived_pl = af.get_nfb_derived_sig(eeg_data_pl, left_alpha_chs, fs, channel_labels, signal_estimator)
right_derived_pl = af.get_nfb_derived_sig(eeg_data_pl, right_alpha_chs, fs, channel_labels, signal_estimator)
left_derived_pr = af.get_nfb_derived_sig(eeg_data_pr, left_alpha_chs, fs, channel_labels, signal_estimator)
right_derived_pr = af.get_nfb_derived_sig(eeg_data_pr, right_alpha_chs, fs, channel_labels, signal_estimator)

fig = px.line()
fig.add_scatter(y=left_derived_pl, name=f"left channels, left probe")
fig.add_scatter(y=right_derived_pl, name="right channels, left probe")
# fig.add_scatter(y=df1['signal_Alpha_Left'][:10000] * 1e-6, name="nfb")
fig.show()

fig = px.line()
fig.add_scatter(y=left_derived_pr, name=f"left channels, right probe")
fig.add_scatter(y=right_derived_pr, name="right channels, right probe")
# fig.add_scatter(y=df1['signal_Alpha_Left'][:10000] * 1e-6, name="nfb")
fig.show()




# --- Get the probe events and MNE raw objects
# Get start of blocks as different types of epochs (1=start, 2=right, 3=left, 4=centre)
df1['protocol_change'] = df1['block_number'].diff()
df1['choice_events'] = df1.apply(lambda row: row.protocol_change if row.block_name == "start" else
                                 row.protocol_change * 2 if row.block_name == "probe_right" else
                                 row.protocol_change * 3 if row.block_name == "probe_left" else
                                 row.protocol_change * 4 if row.block_name == "probe_centre" else 0, axis=1)

# Create the events list for the protocol transitions
probe_events = df1[['choice_events']].to_numpy()
right_probe = 2
left_probe = 3
centre_probe = 4
event_dict = {'right_probe': right_probe, 'left_probe': left_probe, 'centre_probe': centre_probe}

# Drop non eeg data
drop_cols = [x for x in df1.columns if x not in channels]
drop_cols.extend(['MKIDX', 'EOG', 'ECG', "signal_AAI", 'protocol_change', 'choice_events'])
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
# m_raw = m_raw.set_eeg_reference(projection=False) # NOTE: this seems to remove the effect!!!

# Create the stim channel
info = mne.create_info(['STI'], m_raw.info['sfreq'], ['stim'])
stim_raw = mne.io.RawArray(probe_events.T, info)
m_raw.add_channels([stim_raw], force_update_info=True)


# ----- ICA ON BASELINE RAW DATA
# High pass filter
m_high = m_raw.copy()
# Take out the first 10 secs - TODO: figure out if this is needed for everyone
m_high.crop(tmin=10)
m_high.filter(l_freq=1., h_freq=40)
# Drop bad channels
# m_high.drop_channels(['TP9', 'TP10'])
# get baseline data
baseline_raw_data = df1.loc[df1['block_number'] == 2]
baseline_raw_start = baseline_raw_data['sample'].iloc[0] / m_high.info['sfreq']
baseline_raw_end = baseline_raw_data['sample'].iloc[-1] / m_high.info['sfreq']
baseline = m_high.copy()
baseline.crop(tmin=baseline_raw_start, tmax=baseline_raw_end)
# visualise the eog blinks
# eog_evoked = create_eog_epochs(baseline, ch_name=['EOG', 'ECG']).average()
# eog_evoked.apply_baseline(baseline=(None, -0.2))
# eog_evoked.plot_joint()
# do ICA
# baseline.drop_channels(['EOG', 'ECG'])
ica = ICA(n_components=15, max_iter='auto', random_state=97)
ica.fit(baseline)
# m_high.drop_channels(['EOG', 'ECG'])
# ica = ICA(n_components=15, max_iter='auto', random_state=97)
# ica.fit(m_high)
# Visualise
m_high.load_data()
ica.plot_sources(m_high, show_scrollbars=False)
ica.plot_components()
# Set ICA to exclued
ica.exclude = [1]  # ,14]
reconst_raw = m_raw.copy()
ica.apply(reconst_raw)
#-------------------------

#----------get individual peak alpha
iaf_raw = savgol_iaf(m_raw).PeakAlphaFrequency
iaf_bl = savgol_iaf(baseline).PeakAlphaFrequency
#_____------____----____--_-_-_-____-

# Get the epoch object
m_filt = reconst_raw.copy()
m_filt.filter(8, 14, n_jobs=1,  # use more jobs to speed up.
               l_trans_bandwidth=1,  # make sure filter params are the same
               h_trans_bandwidth=1)  # in each band and skip "auto" option.

events = mne.find_events(m_raw, stim_channel='STI')
reject_criteria = dict(eeg=100e-6)

left_chs = ['PO7=1']
right_chs = ['PO8=1']

# - - DO baseline epochs
fig_bl_eo1, aai_baseline_eo1, bl_dataframe_eo1, bl_epochs_eo1 = af.do_baseline_epochs(df1, m_filt, left_chs, right_chs, fig=None, fb_type="active", baseline_name='bl_eo1', block_number=2)
fig_bl_ec1, aai_baseline_ec1, bl_dataframe_ec1, bl_epochs_ec1 = af.do_baseline_epochs(df1, m_filt, left_chs, right_chs, fig=None, fb_type="active", baseline_name='bl_ec1', block_number=4)
fig_bl_eo2, aai_baseline_eo2, bl_dataframe_eo2, bl_epochs_eo2 = af.do_baseline_epochs(df1, m_filt, left_chs, right_chs, fig=None, fb_type="active", baseline_name='bl_eo2', block_number=83)
fig_bl_ec2, aai_baseline_ec2, bl_dataframe_ec2, bl_epochs_ec2 = af.do_baseline_epochs(df1, m_filt, left_chs, right_chs, fig=None, fb_type="active", baseline_name='bl_ec2', block_number=85)


# epochs = mne.Epochs(m_filt, events, event_id=event_dict, tmin=-2, tmax=7, baseline=(-1.5, -0.5),
#                     preload=True, detrend=1, reject=reject_criteria)
epochs = mne.Epochs(m_filt, events, event_id=event_dict, tmin=-2, tmax=7, baseline=None,
                    preload=True, detrend=1)#, reject=reject_criteria)
# epochs.drop([19,22,27,32]) # Drop bads for K's 1st dataset
# epochs.drop([7,17,27,28,29]) # Drop bads for K's 2nd dataset

fig = epochs.plot(events=events)

###### CSP STUFF #############
# Define a monte-carlo cross-validation generator (reduce variance):
scores = []
epochs_data = epochs.get_data()
epochs_train = epochs.copy().crop(tmin=1., tmax=5.)
labels = epochs.events[:, -1] - 2
epochs_data_train = epochs_train.get_data()
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)

# Assemble a classifier
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)

csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
#########################################
# TimeFreq analysis
freqs = np.logspace(*np.log10([6, 35]), num=20)
n_cycles = freqs / 2.  # different number of cycle per frequency
power, itc = tfr_morlet(epochs['left_probe'], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)
power.plot(['PO7'], mode='logratio', title='PO7')
#########################################


probe_left = epochs['left_probe'].average()
probe_right = epochs['right_probe'].average()
# fig2 = probe_left.plot(spatial_colors=True,  picks=['PO7', 'PO8'])
# fig2 = probe_right.plot(spatial_colors=True,  picks=['PO7', 'PO8'])

# # plot topomap#
# fig2 = probe_left.plot_joint()

# TODO Look at PSD of left and right channels for the left and right probes
# TODO compare online AAI (1st run eg) with calculated - these online AAIs aren't smoothed

dataframes = []
# ----Look at the power for the epochs in the left and right channels for left and right probes
e_mean1, e_std1, epoch_pwr1 = af.get_nfb_epoch_power_stats(epochs['left_probe'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names, chs=["PO7=1"])
e_mean2, e_std2, epoch_pwr2 = af.get_nfb_epoch_power_stats(epochs['left_probe'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names,  chs=["PO8=1"])

df_i = pd.DataFrame(dict(left=e_mean1, right=e_mean2))
df_i['section'] = f"left_probe"
dataframes.append(df_i)

fig = go.Figure()
af.plot_nfb_epoch_stats(fig, e_mean1, e_std1, name="left_chs", title="left_probe", color=(230, 20, 20, 1), y_range=[-0.2e-6, 4e-6])
af.plot_nfb_epoch_stats(fig, e_mean2, e_std2, name="right_chs", title="left_probe", color=(20, 20, 220, 1), y_range=[-0.2e-6, 4e-6])
fig.show()
dataframes_aai_cue = []
dataframes_aai = []
aai_nfb_left = (epoch_pwr1 - epoch_pwr2) / (epoch_pwr1 + epoch_pwr2)
df_ix = pd.DataFrame(dict(aai=aai_nfb_left.mean(axis=0)[0]))
df_ix['probe'] = f"left"
dataframes_aai.append(df_ix[2000:9000])
dataframes_aai_cue.append(df_ix[0:2000])
fig1 = go.Figure()
af.plot_nfb_epoch_stats(fig1, aai_nfb_left.mean(axis=0)[0], aai_nfb_left.std(axis=0)[0], name=f"left probe aai",
                     title=f"left probe aai",
                     color=(230, 20, 20, 1), y_range=[-1, 1])


e_mean1, e_std1, epoch_pwr1 = af.get_nfb_epoch_power_stats(epochs['right_probe'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names, chs=["PO7=1"])
e_mean2, e_std2, epoch_pwr2 = af.get_nfb_epoch_power_stats(epochs['right_probe'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names,  chs=["PO8=1"])

df_i = pd.DataFrame(dict(left=e_mean1, right=e_mean2))
df_i['section'] = f"right_probe"
dataframes.append(df_i)

fig = go.Figure()
af.plot_nfb_epoch_stats(fig, e_mean1, e_std1, name="left_chs", title="right_probe", color=(230, 20, 20, 1), y_range=[-0.2e-6, 4e-6])
af.plot_nfb_epoch_stats(fig, e_mean2, e_std2, name="right_chs", title="right_probe", color=(20, 20, 220, 1), y_range=[-0.2e-6, 4e-6])
fig.show()
aai_nfb_right = (epoch_pwr1 - epoch_pwr2) / (epoch_pwr1 + epoch_pwr2)
df_ix = pd.DataFrame(dict(aai=aai_nfb_right.mean(axis=0)[0]))
df_ix['probe'] = f"right"
dataframes_aai.append(df_ix[2000:9000])
dataframes_aai_cue.append(df_ix[0:2000])
af.plot_nfb_epoch_stats(fig1, aai_nfb_right.mean(axis=0)[0], aai_nfb_right.std(axis=0)[0], name=f"right probe aai",
                     title=f"mean aai time course",
                     color=(20, 20, 230, 1), y_range=[-1, 1])


e_mean1, e_std1, epoch_pwr1 = af.get_nfb_epoch_power_stats(epochs['centre_probe'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names, chs=["PO7=1"])
e_mean2, e_std2, epoch_pwr2 = af.get_nfb_epoch_power_stats(epochs['centre_probe'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names,  chs=["PO8=1"])

df_i = pd.DataFrame(dict(left=e_mean1, right=e_mean2))
df_i['section'] = f"centre_probe"
dataframes.append(df_i)

fig = go.Figure()
af.plot_nfb_epoch_stats(fig, e_mean1, e_std1, name="left_chs", title="centre_probe", color=(230, 20, 20, 1), y_range=[-0.2e-6, 4e-6])
af.plot_nfb_epoch_stats(fig, e_mean2, e_std2, name="right_chs", title="centre_probe", color=(20, 20, 220, 1), y_range=[-0.2e-6, 4e-6])
fig.show()
aai_nfb_centre = (epoch_pwr1 - epoch_pwr2) / (epoch_pwr1 + epoch_pwr2)
df_ix = pd.DataFrame(dict(aai=aai_nfb_centre.mean(axis=0)[0]))
df_ix['probe'] = f"centre"
dataframes_aai.append(df_ix[2000:9000])
dataframes_aai_cue.append(df_ix[0:2000])
af.plot_nfb_epoch_stats(fig1, aai_nfb_centre.mean(axis=0)[0], aai_nfb_centre.std(axis=0)[0], name=f"centre probe aai",
                     title=f"mean aai time course",
                     color=(20, 230, 20, 1), y_range=[-1, 1])
fig1.add_vline(x=2000, annotation_text="cvsa start")
fig1.show()

aai_section_df = pd.concat(dataframes_aai)
aai_section_df = aai_section_df.melt(id_vars=['probe'], var_name='side', value_name='data')
px.box(aai_section_df, x='probe', y='data', title="post aai task").show()


# Plot the AAI Boxes
figb = go.Figure()
bl_aai_df1 = pd.DataFrame(dict(bl_eo1=aai_baseline_eo1.mean(axis=0)[0],
                              bl_ec1=aai_baseline_ec1.mean(axis=0)[0]))
bl_aai_df1 = bl_aai_df1.melt(var_name='baseline_type', value_name='data')
figb.add_trace(go.Box(x=bl_aai_df1['baseline_type'], y=bl_aai_df1['data'],
                      notched=True,
                      line=dict(color='blue')))


# TODO - NOTE - NEED TO TAKE OUT BEGINNING OF EPOCHS HERE BECAUSE THEY SHOULDN"T COUNT TO THE MEANS ETC
figb.add_trace(go.Box(y=aai_section_df.loc[aai_section_df['probe'] == 'left']['data'],name='left task') )
figb.add_trace(go.Box(y=aai_section_df.loc[aai_section_df['probe'] == 'right']['data'],name='right task') )
figb.add_trace(go.Box(y=aai_section_df.loc[aai_section_df['probe'] == 'centre']['data'],name='centre task') )

fig.show()
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


# look at online AAI - scalp
left_online_aai = af.get_online_aai(df1, block_name="probe_left")
right_online_aai = af.get_online_aai(df1, block_name="probe_right")
centre_online_aai = af.get_online_aai(df1, block_name="probe_centre")
# fig = go.Figure()
af.plot_nfb_epoch_stats(fig1, left_online_aai['mean'], left_online_aai['std'], name="left_probe online", title="online_aai", color=(220, 20, 20, 1), y_range=[-1, 1])
af.plot_nfb_epoch_stats(fig1, right_online_aai['mean'], right_online_aai['std'], name="right_probe online", title="online_aai", color=(20, 220, 20, 1), y_range=[-1, 1])
af.plot_nfb_epoch_stats(fig1, centre_online_aai['mean'], centre_online_aai['std'], name="centre_probe online", title="online_aai", color=(20, 20, 220, 1), y_range=[-1, 1])


figb.add_trace(go.Box(y=left_online_aai['mean'],name='left online task') )
figb.add_trace(go.Box(y=right_online_aai['mean'],name='right online task') )
figb.add_trace(go.Box(y=centre_online_aai['mean'],name='centre online task') )
figb.update_layout(
    title="online section aai",
    hovermode="x",
)
figb.show()

# look at online AAI - src
left_online_aai = af.get_online_aai(df1, block_name="probe_left", aai_signal_name='signal_AAI_src')
right_online_aai = af.get_online_aai(df1, block_name="probe_right", aai_signal_name='signal_AAI_src')
centre_online_aai = af.get_online_aai(df1, block_name="probe_centre", aai_signal_name='signal_AAI_src')
# fig = go.Figure()
af.plot_nfb_epoch_stats(fig1, left_online_aai['mean'], left_online_aai['std'], name="left_probe online", title="online_aai", color=(220, 20, 20, 1), y_range=[-1, 1])
af.plot_nfb_epoch_stats(fig1, right_online_aai['mean'], right_online_aai['std'], name="right_probe online", title="online_aai", color=(20, 220, 20, 1), y_range=[-1, 1])
af.plot_nfb_epoch_stats(fig1, centre_online_aai['mean'], centre_online_aai['std'], name="centre_probe online", title="online_aai", color=(20, 20, 220, 1), y_range=[-1, 1])

figb.add_trace(go.Box(y=left_online_aai['mean'],name='left online task src') )
figb.add_trace(go.Box(y=right_online_aai['mean'],name='right online task src') )
figb.add_trace(go.Box(y=centre_online_aai['mean'],name='centre online task src') )
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

# TODO: look at channels instead of AAI

# Look at the AAIs of the left, right, and centre plots over the top of eachother


# - Get the power for each epoch

# - Average all the epochs for the PO7 channel - get the mean, and std for each point

# - plot the averaged epochs mean + stds

# LOOK AT ENVELOPE
# probe_left.drop_channels(x for x in channels if x not in ['O1', 'O2', 'P3', 'P4', 'PO7', 'PO8'])
# probe_left.apply_hilbert(picks=['O1', 'O2', 'P3', 'P4', 'PO7', 'PO8'], envelope=True, n_jobs=1, n_fft='auto', verbose=None)
# probe_right.apply_hilbert(picks=['O1', 'O2', 'P3', 'P4', 'PO7', 'PO8'], envelope=True, n_jobs=1, n_fft='auto', verbose=None)
# fig2 = probe_left.plot(spatial_colors=True)
# fig2 = probe_right.plot(spatial_colors=True)
#
# # Look at left and right evoked
# left_chs = ['O1', 'P3', 'PO7']#['O1', 'PO3', 'PO7', 'P1', 'P3', 'P5', 'P7', 'P9', 'PZ', 'P0Z']
# right_chs = ['O2', 'P4', 'PO8']#['O2', 'PO4', 'PO8', 'P2', 'P3', 'P6', 'P7', 'P10', 'PZ', 'P0Z']
# picks = left_chs# + right_chs
# # picks = ['P7', 'PO7', 'O1', 'OZ', 'PO8', 'P8', 'PO3', 'POZ', 'PO4', 'PO8']
# evokeds = dict(left_probe=list(epochs_l['left_probe'].iter_evoked()),
#                right_probe=list(epochs_l['right_probe'].iter_evoked()))
# mne.viz.plot_compare_evokeds(evokeds, combine='mean')#, picks=picks)
#
# # Look at left side vs right side for left probe
# left_ix = mne.pick_channels(probe_left.info['ch_names'], include=right_chs)
# right_ix = mne.pick_channels(probe_left.info['ch_names'], include=left_chs)
# roi_dict = dict(left_ROI=left_ix, right_ROI=right_ix)
# roi_evoked = mne.channels.combine_channels(probe_left, roi_dict, method='mean')
# print(roi_evoked.info['ch_names'])
# roi_evoked.plot()


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

# TODO: ICA (to remove blinks etc)
#   source analysis on epochs


# ---------- SOURCE RECONSTRUCTION---
# Create noise covariance, fwd solution, and inverse operator from first eyes open baseline epochs
noise_cov = mne.compute_covariance(
    bl_epochs_eo1, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=True)
fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, bl_epochs_eo1.info)

# Get the forward solution for the specified source localisation type
fs_dir = fetch_fsaverage(verbose=True)
# --I think this 'trans' is like the COORDS2TRANSFORMATIONMATRIX
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

# m_filt.drop_channels(['ECG', 'EOG'])
# fwd = mne.make_forward_solution(m_filt.info, trans=trans, src=src,
#                                 bem=bem, eeg=True, meg=False, mindist=5.0, n_jobs=1)
fwd = mne.make_forward_solution(bl_epochs_eo1.info, trans=trans, src=src,
                                bem=bem, eeg=True, meg=False, mindist=5.0, n_jobs=1)

# make inverse operator
inverse_operator = make_inverse_operator(
    bl_epochs_eo1.info, fwd, noise_cov, loose=0.2, depth=0.8)
# del fwd

# Get the labels
label_names_lh = ["inferiorparietal-lh", "superiorparietal-lh", "lateraloccipital-lh"]
label_lh = rsf.get_roi_by_name(label_names_lh)
label_names_rh = ["inferiorparietal-rh", "superiorparietal-rh", "lateraloccipital-rh"]
label_rh = rsf.get_roi_by_name(label_names_rh)


# compute inverse solution  for the LEFT SIDE
method = "sLORETA"
snr = 3.
lambda2 = 1. / snr ** 2
stc_lp_lh = apply_inverse_epochs(epochs['left_probe'], inverse_operator, lambda2,
                              method=method, pick_ori=None, verbose=True, label=label_lh)
stc_lp_rh = apply_inverse_epochs(epochs['left_probe'], inverse_operator, lambda2,
                              method=method, pick_ori=None, verbose=True, label=label_rh)

stc_rp_lh = apply_inverse_epochs(epochs['right_probe'], inverse_operator, lambda2,
                              method=method, pick_ori=None, verbose=True, label=label_lh)
stc_rp_rh = apply_inverse_epochs(epochs['right_probe'], inverse_operator, lambda2,
                              method=method, pick_ori=None, verbose=True, label=label_rh)

stc_cp_lh = apply_inverse_epochs(epochs['centre_probe'], inverse_operator, lambda2,
                              method=method, pick_ori=None, verbose=True, label=label_lh)
stc_cp_rh = apply_inverse_epochs(epochs['centre_probe'], inverse_operator, lambda2,
                              method=method, pick_ori=None, verbose=True, label=label_rh)

# stc_rh, stc_lh = af.get_left_right_source_estimates(epochs_active['nfb'])

# repeat above for the baseline
stc_lh_eo1 = apply_inverse_epochs(bl_epochs_eo1, inverse_operator, lambda2,
                              method=method, pick_ori=None, verbose=True, label=label_lh)
stc_rh_eo1 = apply_inverse_epochs(bl_epochs_eo1, inverse_operator, lambda2,
                              method=method, pick_ori=None, verbose=True, label=label_rh)

stc_lh_ec1 = apply_inverse_epochs(bl_epochs_ec1, inverse_operator, lambda2,
                              method=method, pick_ori=None, verbose=True, label=label_lh)
stc_rh_ec1 = apply_inverse_epochs(bl_epochs_ec1, inverse_operator, lambda2,
                              method=method, pick_ori=None, verbose=True, label=label_rh)

stc_lh_eo2 = apply_inverse_epochs(bl_epochs_eo2, inverse_operator, lambda2,
                              method=method, pick_ori=None, verbose=True, label=label_lh)
stc_rh_eo2 = apply_inverse_epochs(bl_epochs_eo2, inverse_operator, lambda2,
                              method=method, pick_ori=None, verbose=True, label=label_rh)

stc_lh_ec2 = apply_inverse_epochs(bl_epochs_ec2, inverse_operator, lambda2,
                              method=method, pick_ori=None, verbose=True, label=label_lh)
stc_rh_ec2 = apply_inverse_epochs(bl_epochs_ec2, inverse_operator, lambda2,
                              method=method, pick_ori=None, verbose=True, label=label_rh)

dataframes_max_source_lp = af.get_source_nfb_section(stc_lp_lh, stc_lp_rh, section_name="left_probe")
dataframes_max_source_rp = af.get_source_nfb_section(stc_rp_lh, stc_rp_rh, section_name="right_probe")
dataframes_max_source_cp = af.get_source_nfb_section(stc_cp_lh, stc_cp_rh, section_name="centre_probe")

dataframes_max_source_bl_eo1 = af.get_source_nfb_section(stc_lh_eo1, stc_rh_eo1, section_name="BL_EO1")
dataframes_max_source_bl_ec1 = af.get_source_nfb_section(stc_lh_ec1, stc_rh_ec1, section_name="BL_EC1")
dataframes_max_source_bl_eo2 = af.get_source_nfb_section(stc_lh_eo2, stc_rh_eo2, section_name="BL_EO2")
dataframes_max_source_bl_ec2 = af.get_source_nfb_section(stc_lh_ec2, stc_rh_ec2, section_name="BL_EC2")

dataframes_max_source_nfb = []
colors = ['blue', 'red']
for s in dataframes_max_source_lp:
    dataframes_max_source_nfb.append(s[2000:9000])
for s in dataframes_max_source_rp:
    dataframes_max_source_nfb.append(s[2000:9000])
for s in dataframes_max_source_cp:
    dataframes_max_source_nfb.append(s[2000:9000])
section_df_nfb_lp = pd.concat(dataframes_max_source_nfb)
section_df_nfb_lp = section_df_nfb_lp.melt(id_vars=['section'], var_name='side', value_name='data')
section_df_nfb_bl_eo1 = dataframes_max_source_bl_eo1[0].melt(id_vars=['section'], var_name='side', value_name='data')
section_df_nfb_bl_ec1 = dataframes_max_source_bl_ec1[0].melt(id_vars=['section'], var_name='side', value_name='data')
section_df_nfb_bl_eo2 = dataframes_max_source_bl_eo2[0].melt(id_vars=['section'], var_name='side', value_name='data')
section_df_nfb_bl_ec2 = dataframes_max_source_bl_ec2[0].melt(id_vars=['section'], var_name='side', value_name='data')
section_df = pd.concat([section_df_nfb_bl_eo1,section_df_nfb_bl_ec1, section_df_nfb_lp, section_df_nfb_bl_eo2, section_df_nfb_bl_ec2,], ignore_index=True)
fig=go.Figure()
for i, side in enumerate(section_df['side'].unique()):
    df_plot = section_df[section_df['side'] == side]
    fig.add_trace(go.Box(x=df_plot['section'], y=df_plot['data'],
                         line=dict(color=colors[i]),
                         name='side=' + side))
# Append the baseline and plot
fig.update_layout(boxmode='group', xaxis_tickangle=1, title='max_source')
fig.show()
