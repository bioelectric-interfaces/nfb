"""
Script to do post hoc source analysis
Steps taken from: Michel, C. M., & Brunet, D. (2019). EEG source imaging: A practical review of the analysis steps. Frontiers in Neurology, 10(APR). https://doi.org/10.3389/fneur.2019.00325
aims:
    1) determine max alpha source in different regions of brain, specifically ocipito-parietal
    2) find time course of sources
    3) compare source locations and time courses between scalp, sham, and source conditions to see lateralisation
    4) compare source locations and time courses between scalp, sham, and source conditions for responders and non responders
"""
import os.path as op

import numpy as np
import matplotlib.pyplot as plt
import plotly_express as px
import plotly.graph_objs as go
import pandas as pd

import mne
from mne.datasets import sample
from mne.channels import make_standard_montage
from mne.viz import plot_sparse_source_estimates
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_raw
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import fetch_fsaverage
from utils.load_results import load_data
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
import analysis.analysis_functions as af

mne.viz.set_3d_backend('pyvista')


#########################
# LOADING DATA
#########################
"""
Get the data from HDF5 Files to MNE data objects
"""
# h5file_scalp = "/Users/christopherturner/Documents/EEG_Data/pilot_202201/ct02/scalp/0-nfb_task_ct02_01-26_16-33-42/experiment_data.h5"

h5file_scalp = "/Users/christopherturner/Documents/EEG_Data/pilot_202201/kk/scalp/0-nfb_task_kk_01-27_18-34-12/experiment_data.h5"
# h5file_sham = ""

df1, m_raw = af.hdf5_to_mne(h5file_scalp)
m_raw, event_dict = af.get_nfb_protocol_change_events(df1, m_raw)

#########################
# PRE-PROCESSING
#########################
# TODO: make this a function (or set of functions) that can be reused across scripts
# TODO: have the option to save off the pre-processed data (for easier processing / reprocessing at later stages) i.e. so selecting bad channels doesn't need to happen every time
"""
Gets rid of artefacts (various different sources of noise)
Can be done automatically but should really be done visually to ensure good results
"""

# # remove projectors
# ssp_projectors = m_raw.info['projs']
# m_raw.del_proj()
#
# # Look at psd
# m_raw.plot_psd(tmax=np.inf, fmax=250, average=True)
# m_raw.plot_psd(tmax=np.inf, fmax=250, spatial_colors=True)
#
# # Look at eye artefacts
# eog_epochs = mne.preprocessing.create_eog_epochs(m_raw, baseline=(-0.5, -0.2))
# eog_epochs.plot_image(combine='mean')
# eog_epochs.average().plot_joint()

#-----------------------
# BAD ELECTRODE DETECTION AND INTERPOLATION
#-----------------------
"""
Remove electrodes whose values go above a certain range (from the resting filtered data)
mne tutorial: https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html

* Do this as early as possible with the raw object - bad channels propegate through the whole analysis

* Detect bad channels from the PSD - if there are some obvious channels that have different / more noise components than the others
* Also detect from looking at the raw plot - if there are some obvious channels that look much different to surrounding channels, (much more noise or are just flat)

* Can also interpolate bad channels (for cross subject analysis to have same data dimentionality)
    If need be interpolation can be done automatically - look here: https://autoreject.github.io/stable/index.html
"""
# m_raw.plot() # Use this to inspect and select any bad channels
# m_raw.interpolate_bads() # This automatically interpolates the bad channels - and clears the 'bads' list after

#-----------------------
# TEMPORAL FILTERING
#-----------------------
"""
Get a specific frequency band and remove non-relavant frequencies

* Use IIR only when you want a steep cutoff (according to: https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html
* Use non-causal to not introduce delays
"""
# Remove low frequency drift
m_raw = m_raw.filter(l_freq=0.1, h_freq=None) # Use 0.1 for evoked data

#non-causal, Infinite Impulse Response (IIR) Butterworth filter of 2nd order
m_filtered = m_raw.copy().filter(l_freq=0.1, h_freq=40) # Use 1hz for non evoked data
alpha_band = (8, 12)# Not sure if we really need both of these - TODO: make sure only relevant things are kept (also do we do a band pass filter like this?)
m_alpha = m_raw.copy().filter(l_freq=alpha_band[0], h_freq=alpha_band[1])


#-----------------------
# MARKING BAD SPANS
#-----------------------
"""
This can be done manually or automatically
This is really only needed if there are some specifically bad sections in the data - if they interfere with epochs they can be taken out then
"""

#-----------------------
# ICA
#-----------------------
"""
do ica correction to remove artefacts
do this on the baseline data and apply to the entire raw dataset
"""
# # ----- ICA ON BASELINE RAW DATA
# # High pass filter
# m_high = m_raw.copy()
# # Take out the first 10 secs - TODO: figure out if this is needed for everyone
# m_high.crop(tmin=10)
# m_high.filter(l_freq=1., h_freq=40)
# # get baseline data
# baseline_raw_data = df1.loc[df1['block_name'] == 'baseline']
# baseline_raw_start = baseline_raw_data['sample'].iloc[0] / fs
# baseline_raw_end = baseline_raw_data['sample'].iloc[-1] / fs
# baseline = m_high.copy()
# baseline.crop(tmin=baseline_raw_start, tmax=baseline_raw_end)
# # visualise the eog blinks
# # eog_evoked = create_eog_epochs(baseline, ch_name=['EOG', 'ECG']).average()
# # eog_evoked.apply_baseline(baseline=(None, -0.2))
# # eog_evoked.plot_joint()
# # do ICA
# baseline.drop_channels(['EOG', 'ECG'])
# ica = ICA(n_components=15, max_iter='auto', random_state=97)
# ica.fit(baseline)
# # Visualise
# m_high.load_data()
# ica.plot_sources(m_high, show_scrollbars=False)
# ica.plot_components()
# # Set ICA to exclued
# ica.exclude = [1, 11, 12, 13]  # ,14]
# reconst_raw = m_raw.copy()
# ica.apply(reconst_raw)


#-----------------------
# DOWN SAMPLING
#-----------------------
"""
Down sample the data to 4* highest frequency after filtering
Best practice - (to avoid issues with messing with timings and also issues with edge effects doing this after epoching)
1. low-pass filter the Raw data at or below 1/3 of the desired sample rate, then
2. decimate the data after epoching, by either passing the decim parameter to the Epochs constructor, or using the decimate() method after the Epochs have been created.

-OR- (if can't do best practice) 
Do this AFTER epoching (as it jitters the timings)
"""
# TODO: Should this be used instead of band pass filtering?
# current_sfreq = m_raw.info['sfreq']
# desired_sfreq = alpha_band[1]*4  # Hz
# decim = np.round(current_sfreq / desired_sfreq).astype(int)
# obtained_sfreq = current_sfreq / decim
# lowpass_freq = obtained_sfreq / 3.
#
# raw_filtered = m_raw.copy().filter(l_freq=None, h_freq=lowpass_freq)

#-----------------------
# EPOCHING
#-----------------------
"""
epoch the data in terms of events
for the nfb task - events are the following
    * onset of nfb protocol (duration - till the end of the nfb task)
    * visual probe in delay protocol (left and right)
detect and eliminate bad epochs in this stage
"""

# get baseline length for epoching in equal size
baseline_start = df1.loc[df1['block_name'] == 'baseline']['sample'].iloc[0] / m_alpha.info['sfreq']
baseline_end = df1.loc[df1['block_name'] == 'baseline']['sample'].iloc[-1] / m_alpha.info['sfreq']
baseline_alpha = m_alpha.copy().crop(tmin=baseline_start, tmax=baseline_end)

# Get baseline epochs of same length as other epochs (6 seconds)
baseline_epochs = mne.make_fixed_length_epochs(baseline_alpha, duration=6, preload=False)

left_chs = ["CP5=1","P5=1","O1=1"]
right_chs = ["CP6=1","P6=1","O2=1"]

# look at the alpha power for the baseline for left and right channels
e_mean1, e_std1, e_pwr1 = af.get_nfb_epoch_power_stats(baseline_epochs, fband=(8, 12), fs=1000, channel_labels=baseline_epochs.info.ch_names, chs=left_chs)
e_mean2, e_std2, e_pwr2 = af.get_nfb_epoch_power_stats(baseline_epochs, fband=(8, 12), fs=1000, channel_labels=baseline_epochs.info.ch_names,  chs=right_chs)
fig = go.Figure()
af.plot_nfb_epoch_stats(fig, e_mean1, e_std1, name=";".join(left_chs), title="baesline epoch power", color=(230,20,20,1), y_range=[0, 10e-6])
af.plot_nfb_epoch_stats(fig, e_mean2, e_std2, name=";".join(right_chs), title="baesline epoch power", color=(20,20,230,1), y_range=[0, 10e-6])
fig.show()

# get calculated AAI for baseline (left-right/left+right)
aai_baseline = (e_pwr1-e_pwr2)/(e_pwr1+e_pwr2)
fig = go.Figure()
af.plot_nfb_epoch_stats(fig, aai_baseline.mean(axis=0)[0], aai_baseline.std(axis=0)[0], name="aai", title="epoch power", color=(230,20,20,1), y_range=[-0.7, 0.7])
fig.show()

bd = pd.DataFrame(dict(left_mean=e_mean1, right_mean=e_mean2)).melt(var_name="data")
px.box(bd, y='value', color='data', title="total nfb epochs").show()

## Epoch the other sections
events = mne.find_events(m_raw, stim_channel='STI')
reject_criteria = dict(eeg=1000e-6)

epochs = mne.Epochs(m_alpha, events, event_id=event_dict, tmin=-1, tmax=5, baseline=None,
                    preload=True, detrend=1, reject=reject_criteria) # TODO: make sure baseline params correct (using white fixation cross as baseline: (None, -1)

# look at the alpha power for the different sections (increase left alpha) for left and right channels
for eid, k in epochs.event_id.items():
    e_mean1, e_std1, e_pwr1 = af.get_nfb_epoch_power_stats(epochs[eid], fband=(8, 12), fs=1000,
                                                   channel_labels=epochs.info.ch_names, chs=left_chs)
    e_mean2, e_std2, e_pwr2 = af.get_nfb_epoch_power_stats(epochs[eid], fband=(8, 12), fs=1000,
                                                   channel_labels=epochs.info.ch_names, chs=right_chs)
    fig = go.Figure()
    af.plot_nfb_epoch_stats(fig, e_mean1, e_std1, name=";".join(left_chs), title=eid, color=(230, 20, 20, 1), y_range=[0, 5e-6])
    af.plot_nfb_epoch_stats(fig, e_mean2, e_std2, name=";".join(right_chs), title=eid, color=(20, 20, 230, 1), y_range=[0, 5e-6])
    fig.show()

    # get calculated AAI for nfb (left-right/left+right)
    if eid == 'nfb':
        aai_nfb = (e_pwr1-e_pwr2)/(e_pwr1+e_pwr2)
        fig1 = go.Figure()
        af.plot_nfb_epoch_stats(fig1, aai_nfb.mean(axis=0)[0], aai_nfb.std(axis=0)[0], name=f"aai", title=f"{eid} aai", color=(230,20,20,1), y_range=[-1, 1])
        fig1.show()


#----------=======-------=======-------=========
# # compare with online aai - IT LOOKS THE SAME IF THE START OF THE NFB EPOCHS ABOVE IS 0 - this is just smoothed!
# drop_cols = [x for x in df1.columns if x not in ['signal_AAI', 'block_number', 'block_name', "sample"]]
# aai_df = df1.drop(columns=drop_cols)
# aai_df = aai_df[aai_df['block_name'] == "NFB"]
# aai_df = aai_df.drop(columns='block_name')
# signal_aai = []
# for x in aai_df.block_number.unique():
#     signal_aai.append(aai_df.loc[aai_df.block_number == x].signal_AAI.reset_index(drop=True))
# # fig = go.Figure()
# af.plot_nfb_epoch_stats(fig1, pd.concat(signal_aai, axis=1).mean(axis=1), 0, name=f"aai",
#                             title=f"aai", color=(230, 20, 20, 1))
# fig1.show()

## double checking the first epoch is what I thnk it is - SEEMS IT IS!!!! (so is the baseline!!)
# nfb1_start = aai_df.loc[aai_df.block_number == 8]['sample'].iloc[0] / m_alpha.info['sfreq']
# nfb1_end = aai_df.loc[aai_df.block_number == 8]['sample'].iloc[-1] / m_alpha.info['sfreq']
# nfb1_alpha = m_alpha.copy().crop(tmin=nfb1_start, tmax=nfb1_end)
# nfb1_epochs = mne.make_fixed_length_epochs(nfb1_alpha, duration=5, preload=False)
# e_mean1, e_std1, e_pwr1 = af.get_nfb_epoch_power_stats(nfb1_epochs, fband=(8, 12), fs=1000, channel_labels=nfb1_epochs.info.ch_names, chs=left_chs)
# e_mean2, e_std2, e_pwr2 = af.get_nfb_epoch_power_stats(nfb1_epochs, fband=(8, 12), fs=1000, channel_labels=nfb1_epochs.info.ch_names,  chs=right_chs)
# fig = go.Figure()
# af.plot_nfb_epoch_stats(fig, e_mean1, e_std1, name=";".join(left_chs), title="nfb1 epoch power", color=(230,20,20,1), y_range=[0, 10e-6])
# af.plot_nfb_epoch_stats(fig, e_mean2, e_std2, name=";".join(right_chs), title="nfb1 epoch power", color=(20,20,230,1), y_range=[0, 10e-6])
# fig.show()
#
# # get calculated AAI for baseline (left-right/left+right)
# aai_nfb1= (e_pwr1-e_pwr2)/(e_pwr1+e_pwr2)
# fig = go.Figure()
# af.plot_nfb_epoch_stats(fig, aai_nfb1.mean(axis=0)[0], aai_nfb1.std(axis=0)[0], name="nfb1 aai", title="epoch power", color=(230,20,20,1), y_range=[-0.7, 0.7])
# af.plot_nfb_epoch_stats(fig, signal_aai[0], 0, name="nfb1 aai ol", title="epoch power", color=(30,120,20,1), y_range=[-0.7, 0.7])
# fig.show()
#----------=======-------=======-------=========


# Look at the nfb epochs in quartered time sections
step = int(len(epochs['nfb'])/4)
section_data = {}
dataframes = []
dataframes_aai = []
for i, x in enumerate(range(0, 100, step)):
    e_mean1, e_std1, e_pwr1 = af.get_nfb_epoch_power_stats(epochs['nfb'][x:x+step], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names, chs=left_chs)
    e_mean2, e_std2, e_pwr2 = af.get_nfb_epoch_power_stats(epochs['nfb'][x:x+step], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names,  chs=right_chs)
    fig = go.Figure()
    af.plot_nfb_epoch_stats(fig, e_mean1, e_std1, name=";".join(left_chs), title=f"nfb ({x}:{x+step})", color=(230, 20, 20, 1), y_range=[0, 5e-6])
    af.plot_nfb_epoch_stats(fig, e_mean2, e_std2, name=";".join(right_chs), title=f"nfb ({x}:{x+step})", color=(20, 20, 230, 1), y_range=[0, 5e-6])
    fig.show()
    # get the means for box plotting
    df_i = pd.DataFrame(dict(left=e_mean1, right=e_mean2))
    df_i['section'] = f"Q{i+1}"
    dataframes.append(df_i)

    # Get the AAIs of these sections
    aai_nfb = (e_pwr1 - e_pwr2) / (e_pwr1 + e_pwr2)
    fig1 = go.Figure()
    af.plot_nfb_epoch_stats(fig1, aai_nfb.mean(axis=0)[0], aai_nfb.std(axis=0)[0], name=f"aai", title=f"nfb ({x}:{x+step}) aai",
                            color=(230, 20, 20, 1), y_range=[-1, 1])
    fig1.show()
    # get the aais for box plotting
    df_ix = pd.DataFrame(dict(aai=aai_nfb.mean(axis=0)[0]))
    df_ix['section'] = f"Q{i+1}"
    dataframes_aai.append(df_ix)

# Plot the boxes of the left and right powers
section_df = pd.concat(dataframes)
section_df = section_df.melt(id_vars=['section'], var_name='side', value_name='data')
px.box(section_df, x='section', y='data', color='side', title="sectioned nfb epochs").show()

# plot the boxes of the aais
aai_section_df = pd.concat(dataframes_aai)
aai_section_df = aai_section_df.melt(id_vars=['section'], var_name='side', value_name='data')
px.box(aai_section_df, x='section', y='data', title="sectioned aai").show()

# -DECIMATE STUFF>>>>
# events = mne.find_events(raw_filtered)
# epochs = mne.Epochs(raw_filtered, events, decim=decim)
#
# print('desired sampling frequency was {} Hz; decim factor of {} yielded an '
#       'actual sampling frequency of {} Hz.'
#       .format(desired_sfreq, decim, epochs.info['sfreq']))

# Epoch all NFB trials (check what bagherzadeh does) - then average the source time course of these across participants

# USE 'reject' to automatically reject bad epochs (over certain peak-peak amplitudes)
#    reject_tmin and reject_tmax can also be used to determine the time frame to reject epochs (default whole epoch)
# ->>>>>>>>>>>>>>>>>>>

