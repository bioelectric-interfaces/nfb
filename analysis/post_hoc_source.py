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
h5file = "/Users/christopherturner/Documents/EEG_Data/pilot_202201/ct02/scalp/0-nfb_task_ct02_01-26_16-33-42/experiment_data.h5"

# Put data in pandas data frame
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

# Drop non eeg data
drop_cols = [x for x in df1.columns if x not in channels]
drop_cols.extend(['MKIDX'])
eeg_data = df1.drop(columns=drop_cols)

# Rescale the data (units are microvolts - i.e. x10^-6
eeg_data = eeg_data * 1e-6

# create an MNE info - set types appropriately
m_info = mne.create_info(ch_names=list(eeg_data.columns), sfreq=fs, ch_types=['eeg' if ch not in ['ECG', 'EOG'] else 'eog' for ch in list(eeg_data.columns)])

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


#----------------------------------------
# Get nfb trials as epochs from hdf5 data
#----------------------------------------
df1['protocol_change'] = df1['block_number'].diff()
df1['p_change_events'] =  df1.apply(lambda row: row.protocol_change if row.block_name == "NFB" else
                                 row.protocol_change * 2 if row.block_name == "fc_w" else
                                 row.protocol_change * 3 if row.block_name == "fc_b" else
                                 row.protocol_change * 4 if row.block_name == "delay" else
                                 row.protocol_change * 5 if row.block_name == "Input" else 0, axis=1)


# Create the events list for the protocol transitions
probe_events = df1[['p_change_events']].to_numpy()
event_dict = {'nfb': 1, 'fc_w': 2, 'fc_b': 3, 'delay': 4, 'Input': 5}

# Create the stim channel
info = mne.create_info(['STI'], m_raw.info['sfreq'], ['stim'])
stim_raw = mne.io.RawArray(probe_events.T, info)
m_raw.add_channels([stim_raw], force_update_info=True)

#########################
# PRE-PROCESSING
#########################
# TODO: make this a function (or set of functions) that can be reused across scripts
# TODO: have the option to save off the pre-processed data (for easier processing / reprocessing at later stages) i.e. so selecting bad channels doesn't need to happen every time
"""
Gets rid of artefacts (various different sources of noise)
Can be done automatically but should really be done visually to ensure good results
"""

# remove projectors
ssp_projectors = m_raw.info['projs']
m_raw.del_proj()

# Look at psd
m_raw.plot_psd(tmax=np.inf, fmax=250, average=True)
m_raw.plot_psd(tmax=np.inf, fmax=250, spatial_colors=True)

# Look at eye artefacts
eog_epochs = mne.preprocessing.create_eog_epochs(m_raw, baseline=(-0.5, -0.2))
eog_epochs.plot_image(combine='mean')
eog_epochs.average().plot_joint()

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
m_raw.plot() # Use this to inspect and select any bad channels
m_raw.interpolate_bads() # This automatically interpolates the bad channels - and clears the 'bads' list after

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
alpha_band = (8, 14)# Not sure if we really need both of these - TODO: make sure only relevant things are kept (also do we do a band pass filter like this?)
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
## Epoch and downsample in one go
events = mne.find_events(m_raw, stim_channel='STI')
reject_criteria = dict(eeg=1000e-6)

epochs = mne.Epochs(m_alpha, events, event_id=event_dict, tmin=-1, tmax=5, baseline=None,
                    preload=True, detrend=1, reject=reject_criteria) # TODO: make sure baseline params correct (using white fixation cross as baseline: (None, -1)

# look at the alpha power for the nfb trials (increase left alpha) for left and right channels
e_mean1, e_std1 = af.get_nfb_epoch_power_stats(epochs['nfb'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names, chs=["PO7=1"])
e_mean2, e_std2 = af.get_nfb_epoch_power_stats(epochs['nfb'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names,  chs=["PO8=1"])
af.plot_nfb_epoch_stats(e_mean1, e_std1, e_mean2, e_std2, name1="left_chs", name2="right_chs", title="nfb")

# look at the alpha power for the white fixation dot trials (increase left alpha) for left and right channels
e_mean1, e_std1 = af.get_nfb_epoch_power_stats(epochs['fc_w'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names, chs=["PO7=1"])
e_mean2, e_std2 = af.get_nfb_epoch_power_stats(epochs['fc_w'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names,  chs=["PO8=1"])
af.plot_nfb_epoch_stats(e_mean1, e_std1, e_mean2, e_std2, name1="left_chs", name2="right_chs", title="fc_w")

e_mean1, e_std1 = af.get_nfb_epoch_power_stats(epochs['fc_b'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names, chs=["PO7=1"])
e_mean2, e_std2 = af.get_nfb_epoch_power_stats(epochs['fc_b'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names,  chs=["PO8=1"])
af.plot_nfb_epoch_stats(e_mean1, e_std1, e_mean2, e_std2, name1="left_chs", name2="right_chs", title="fc_b")

e_mean1, e_std1 = af.get_nfb_epoch_power_stats(epochs['delay'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names, chs=["PO7=1"])
e_mean2, e_std2 = af.get_nfb_epoch_power_stats(epochs['delay'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names,  chs=["PO8=1"])
af.plot_nfb_epoch_stats(e_mean1, e_std1, e_mean2, e_std2, name1="left_chs", name2="right_chs", title="delay")

e_mean1, e_std1 = af.get_nfb_epoch_power_stats(epochs['Input'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names, chs=["PO7=1"])
e_mean2, e_std2 = af.get_nfb_epoch_power_stats(epochs['Input'], fband=(8, 14), fs=1000, channel_labels=epochs.info.ch_names,  chs=["PO8=1"])
af.plot_nfb_epoch_stats(e_mean1, e_std1, e_mean2, e_std2, name1="left_chs", name2="right_chs", title="Input")

# events = mne.find_events(raw_filtered)
# epochs = mne.Epochs(raw_filtered, events, decim=decim)
#
# print('desired sampling frequency was {} Hz; decim factor of {} yielded an '
#       'actual sampling frequency of {} Hz.'
#       .format(desired_sfreq, decim, epochs.info['sfreq']))

# Epoch all NFB trials (check what bagherzadeh does) - then average the source time course of these across participants

# USE 'reject' to automatically reject bad epochs (over certain peak-peak amplitudes)
#    reject_tmin and reject_tmax can also be used to determine the time frame to reject epochs (default whole epoch)


