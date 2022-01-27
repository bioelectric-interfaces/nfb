"""
File to analyse the ERP data from the probe
this does:
    A) look at ERP power in left and right side
    B) look at source location for left and right ERPs
"""
import sys
import os

from philistine.mne import write_raw_brainvision

from pynfb.helpers.roi_spatial_filter import get_roi_filter

sys.path.append(f"{os.getcwd()}")

import matplotlib.pyplot as plt
import numpy as np
from mne.channels import make_standard_montage
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw, apply_inverse
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)

from utils.load_results import load_data
from pynfb.helpers import roi_spatial_filter as rsf
import glob
import pandas as pd
import plotly.express as px
from scipy.signal import butter, lfilter, freqz
import mne

# TODO: LOOK AT THIS FOR PERMUTATION TESTING OF ERPS
# https://mne.tools/stable/auto_tutorials/stats-source-space/20_cluster_1samp_spatiotemporal.html#sphx-glr-auto-tutorials-stats-source-space-20-cluster-1samp-spatiotemporal-py

mne.viz.set_3d_backend('pyvista')

# # ------ Get data files
# data_directory = "/Users/christopherturner/Documents/EEG_Data/pilot_202201" # This is the directory where all participants are in
#
# # get participants
# participants = next(os.walk(data_directory))[1]
#
# # Get scalp, sham, source data for each participant
# experiment_dirs = {}
# for p in participants:
# # TODO: fix data file structure (when have a solid test setup) - maybe include 'sc', 'so', 'sh' in the data directory names
# #       and allocate this way - this way don't have to sort into separate folders. - if do 2 tasks per session, then also don't have to copy
#     experiment_dirs[p] = {}
#     experiment_dirs[p]["scalp"] = next(os.walk(os.path.join(data_directory, p, "scalp")))[1]
#     experiment_dirs[p]["source"] = next(os.walk(os.path.join(data_directory, p, "source")))[1]
#     experiment_dirs[p]["sham"] = next(os.walk(os.path.join(data_directory, p, "sham")))[1]

# experiment_data = []
# for participant, participant_dirs in experiment_dirs.items():
#     participant_data = {"participant_id": participant, "session_data": []}
#     if participant == "ct02":
#         for session, session_dirs in participant_dirs.items():
#             session_data = {}
#             for task_dir in session_dirs:
#                 if "nfb" in task_dir:
#                     task_data = {}
# h5file = "/Users/christopherturner/Documents/EEG_Data/system_testing/raw_data/large_probe_test_01-26_17-17-23/experiment_data.h5"
# h5file = "/Users/christopherturner/Documents/EEG_Data/system_testing/raw_data/large_probe_test_noprobe_01-26_17-28-51/experiment_data.h5"
h5file = "/Users/christopherturner/Documents/EEG_Data/system_testing/raw_data/screen_change_test_01-26_17-38-42/experiment_data.h5"

# Put data in pandas data frame
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

# get the protocol data and average the AAI for each protocol
# protocol_data = af.get_protocol_data(df1, channels=channels, p_names=p_names, eog_filt=False)

# probe protocols = 'delay<n>' # TODO: make this work for all protocols
pass

# -----
# Get the samples where the probes start
# inspection = df1.loc[df1.block_name == 'delay']
inspection = df1.loc[df1.block_name == 'probe']
# right probes
probes = inspection.loc[inspection.probe.isin([1, 2])]
block_no = 0
# for index, row in probes.iterrows():
    # if row['block_number'] != block_no and row['block_name'] == 'delay':
    # if row['block_number'] != block_no and row['block_name'] == 'probe':
    #     df1['probe_events'].iloc[row['sample']] = row['probe']
    #     block_no = row['block_number']

# -----
# Get the probe events just at every 5 seconds so they don't align with probes (use in no-probe data)
# df1['probe_events'] = 0
# idx = 5000
# for index, row in df1.iterrows():
#     if idx < len(df1)-10000:
#         df1['probe_events'].iloc[idx] = 1
#         df1['probe_events'].iloc[idx+1000] = 2
#         idx += 1500


# Get probe at block transitions (for black/white screen transitions)
df1['protocol_change'] = df1['block_number'].diff()
df1['probe_events'] = df1.apply(lambda row: row.protocol_change*2 if row.block_name == "probe" else (row.protocol_change if row.block_name == "wait" else 0), axis=1)
# df1['probe_events'] = df1.apply(lambda row: row.protocol_change if row.block_name == "wait" else 0, axis=1)



# ------
# Get all samples where choice protocol starts
# Put the transition (event) data back in the original dataframe
# df1['protocol_change'] = df1['block_number'].diff()
# df1['choice_events'] = df1.apply(lambda row: row.protocol_change if row.block_name == "Input" else 0, axis = 1)


# ----Get just a single epoch (when the screen transitions for the first time)---
# first_transition = df1.loc[df1.choice_events == 1].iloc[0].loc['sample']
# df1 = df1.loc[first_transition - 500:first_transition + 500, :]
#--------------


# Create the events list for the protocol transitions
probe_events = df1[['probe_events']].to_numpy()
right_probe = 1
left_probe = 2
event_dict = {'right_probe': right_probe, 'left_probe': left_probe}

# Drop non eeg data
drop_cols = [x for x in df1.columns if x not in channels]
drop_cols.extend(['MKIDX', 'probe_events'])
eeg_data = df1.drop(columns=drop_cols)




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

# Create channel types
# m_raw.info['ch_types'] = ['eog' if tp in ['EOG', 'ECG'] else 'stim' if tp == 'STI' else 'eeg' for tp in
#  m_raw.info['ch_names']]

# Create the stim channel
info = mne.create_info(['STI'], m_raw.info['sfreq'], ['stim'])
stim_raw = mne.io.RawArray(probe_events.T, info)
m_raw.add_channels([stim_raw], force_update_info=True)

# Save EEG with stim data as a brainvision file
# brainvision_file = os.path.join(os.getcwd(), f'{p}_{session}_{task_dir}-bv.vhdr')
# write_raw_brainvision(m_raw, brainvision_file, events=True)

# set the reference to average
m_raw.set_eeg_reference(projection=True)
# fig = m_raw.plot(scalings={"eeg":10})

# Remove bads manually TODO: do this automatically
# m_raw.info['bads'] = ['TP9', 'TP10', 'FT7', 'FT8', 'T7', 'OZ'] # For Ksenia's large dataset
# m_raw.info['bads'] = ['TP7', 'TP8', 'TP9', 'TP10', 'AF7', 'AF8', 'T7', 'T8']


# ----- ICA ON BASELINE RAW DATA
# High pass filter
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
# ica.exclude = [1,11,12,13]#,14]
# reconst_raw = m_raw.copy()
# ica.apply(reconst_raw)

reconst_raw = m_raw.copy()
# # low pass at 40hz
m_filt = reconst_raw.copy()
# m_filt.filter(l_freq=0, h_freq=10)
m_filt.filter(l_freq=1., h_freq=25)
# Take off the first 10 secs
m_filt.crop(tmin=10)

# # epoch the data
events = mne.find_events(reconst_raw, stim_channel='STI')
picks = [c for c in m_info.ch_names if c not in ['EOG', 'ECG']]
epochs = mne.Epochs(m_filt, events, event_id=event_dict, tmin=-0.1, tmax=0.4,
                    preload=True, picks=picks, detrend=1)
fig = epochs.plot(events=events, scalings={"eeg":10})
# epochs.apply_baseline((None, 0))

# average epochs
probe_left = epochs['left_probe'].average()
probe_right = epochs['right_probe'].average()
fig2 = probe_left.plot(spatial_colors=True)
fig2 = probe_right.plot(spatial_colors=True)
# choice_transition.plot_joint()

# plot topomap
probe_left.plot_topomap(times=[-0.1, 0.1, 0.4], average=0.05)
probe_right.plot_topomap(times=[-0.1, 0.1, 0.4], average=0.05)
probe_left.plot_joint(title="left")
probe_right.plot_joint(title="right")

# Look at left and right evoked
left_chs = ['O1', 'PO3', 'PO7', 'P1', 'P3', 'P5', 'P7', 'P9', 'PZ', 'P0Z' ]
right_chs = ['O2', 'PO4', 'PO8', 'P2', 'P3', 'P6', 'P7', 'P10', 'PZ', 'P0Z']
picks = left_chs + right_chs
# picks = ['P7', 'PO7', 'O1', 'OZ', 'PO8', 'P8', 'PO3', 'POZ', 'PO4', 'PO8']
evokeds = dict(left_probe=list(epochs['left_probe'].iter_evoked()),
               right_probe=list(epochs['right_probe'].iter_evoked()))
mne.viz.plot_compare_evokeds(evokeds, combine='mean', picks=picks)
pass

# Look at left side vs right side for left probe
left_ix = mne.pick_channels(probe_left.info['ch_names'], include=right_chs)
right_ix = mne.pick_channels(probe_left.info['ch_names'], include=left_chs)
roi_dict = dict(left_ROI=left_ix, right_ROI=right_ix)
roi_evoked = mne.channels.combine_channels(probe_left, roi_dict, method='mean')
print(roi_evoked.info['ch_names'])
roi_evoked.plot()

# Look at left side vs right side for right probe
left_ix = mne.pick_channels(probe_right.info['ch_names'], include=right_chs)
right_ix = mne.pick_channels(probe_right.info['ch_names'], include=left_chs)
roi_dict = dict(left_ROI=left_ix, right_ROI=right_ix)
roi_evoked = mne.channels.combine_channels(probe_right, roi_dict, method='mean')
print(roi_evoked.info['ch_names'])
roi_evoked.plot()





# ---------- SOURCE RECONSTRUCTION---
noise_cov = mne.compute_covariance(
    epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=True)
fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, reconst_raw.info)

# Get the forward solution for the specified source localisation type
fs_dir = fetch_fsaverage(verbose=True)
# --I think this 'trans' is like the COORDS2TRANSFORMATIONMATRIX
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

m_filt.drop_channels(['ECG', 'EOG'])
# fwd = mne.make_forward_solution(m_filt.info, trans=trans, src=src,
#                                 bem=bem, eeg=True, meg=False, mindist=5.0, n_jobs=1)
fwd = mne.make_forward_solution(probe_left.info, trans=trans, src=src,
                                bem=bem, eeg=True, meg=False, mindist=5.0, n_jobs=1)

# make inverse operator
inverse_operator = make_inverse_operator(
    probe_left.info, fwd, noise_cov, loose=0.2, depth=0.8)
# del fwd

# compute inverse solution of whole brain
method = "sLORETA"
snr = 3.
lambda2 = 1. / snr ** 2
stc, residual = apply_inverse(probe_left, inverse_operator, lambda2,
                              method=method, pick_ori=None,
                              return_residual=True, verbose=True)

# plot the time course of the peak source
vertno_max_idx, time_max = stc.get_peak(hemi=None,vert_as_index=True)
fig, ax = plt.subplots()
ax.plot(1e3 * stc.times, stc.data[vertno_max_idx])
ax.set(xlabel='time (ms)', ylabel='%s value' % method)


# look at the peak
vertno_max, time_max = stc.get_peak(hemi=None)

surfer_kwargs = dict(
    hemi='both',
    clim='auto', views='lateral',  # clim=dict(kind='value', lims=[8, 12, 15])
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10,
    surface='Pial', transparent=True, alpha=0.9, colorbar=True, show_traces=True)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'left probe', 'title',
               font_size=14)

stc, residual = apply_inverse(probe_right, inverse_operator, lambda2,
                              method=method, pick_ori=None,
                              return_residual=True, verbose=True)

# plot the time course of the peak source
vertno_max_idx, time_max = stc.get_peak(hemi=None,vert_as_index=True)
fig, ax = plt.subplots()
ax.plot(1e3 * stc.times, stc.data[vertno_max_idx])
ax.set(xlabel='time (ms)', ylabel='%s value' % method)


# look at the peak
vertno_max, time_max = stc.get_peak(hemi=None)

surfer_kwargs = dict(
    hemi='both',
    clim='auto', views='lateral',  # clim=dict(kind='value', lims=[8, 12, 15])
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10,
    surface='Pial', transparent=True, alpha=0.9, colorbar=True, show_traces=True)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'right probe', 'title',
               font_size=14)



# Get inverse solution of specific labels (left and right)
label_names_lh = ["inferiorparietal-lh", "superiorparietal-lh", "lateraloccipital-lh"]
label_lh = rsf.get_roi_by_name(label_names_lh)
label_names_rh = ["inferiorparietal-rh", "superiorparietal-rh", "lateraloccipital-rh"]
label_rh = rsf.get_roi_by_name(label_names_rh)

stc_lh, residual_lh = apply_inverse(probe_left, inverse_operator, lambda2,
                                    method=method, pick_ori=None,
                                    return_residual=True, verbose=True, label=label_lh)
stc_rh, residual_rh = apply_inverse(probe_right, inverse_operator, lambda2,
                                    method=method, pick_ori=None,
                                    return_residual=True, verbose=True, label=label_rh)

vertno_max, time_max = stc_lh.get_peak(hemi=None)
surfer_kwargs = dict(
    hemi='both',
    clim='auto', views='lateral',  # clim=dict(kind='value', lims=[8, 12, 15])
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10,
    surface='Pial', transparent=True, alpha=0.9, colorbar=True, show_traces=True)
brain = stc_lh.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'left', 'title',
               font_size=14)
brain.add_label(label_lh)

vertno_max, time_max = stc_rh.get_peak(hemi=None)
surfer_kwargs = dict(
    hemi='both',
    clim='auto', views='lateral',  # clim=dict(kind='value', lims=[8, 12, 15])
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10,
    surface='Pial', transparent=True, alpha=0.9, colorbar=True, show_traces=True)
brain = stc_rh.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'right', 'title',
               font_size=14)
brain.add_label(label_rh)



# Do a matplotlib to hold the brain plot in place!!
plt.plot(1e3 * stc.times, stc.data[::100, :].T)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
plt.title('all')
plt.show()

plt.plot(1e3 * stc_lh.times, stc_lh.data[::100, :].T)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
plt.title('left')
plt.show()

plt.plot(1e3 * stc_rh.times, stc_rh.data[::100, :].T)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
plt.title('right')
plt.show()
pass

# use NFBLab inverse operator
inverse_operator_nfb = mne.minimum_norm.make_inverse_operator(m_filt.info, fwd, noise_cov,
                                                          fixed=True) # NOTE: the difference with the MNE tutorial methods is the fixed oritentations - when applying the inverse in this case the result isn't normed (so you get signed source activity)
stc_nfblab_left, residual = apply_inverse(probe_left, inverse_operator_nfb, lambda2,
                                     method=method, pick_ori=None,
                                     return_residual=True, verbose=True) # This is the same as the way above except more or less the absolute value - IS THIS WHAT WE NEED???

vertno_max, time_max = stc_nfblab_left.get_peak(hemi=None)
surfer_kwargs = dict(
    hemi='both',
    clim='auto', views='lateral',  # clim=dict(kind='value', lims=[8, 12, 15])
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10,
    surface='Pial', transparent=True, alpha=0.9, colorbar=True, show_traces=True)
brain = stc_nfblab_left.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'NFBLab method left', 'title',
               font_size=14)

stc_nfblab_right, residual = apply_inverse(probe_right, inverse_operator_nfb, lambda2,
                                     method=method, pick_ori=None,
                                     return_residual=True, verbose=True) # This is the same as the way above except more or less the absolute value - IS THIS WHAT WE NEED???

vertno_max, time_max = stc_nfblab_right.get_peak(hemi=None)
surfer_kwargs = dict(
    hemi='both',
    clim='auto', views='lateral',  # clim=dict(kind='value', lims=[8, 12, 15])
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10,
    surface='Pial', transparent=True, alpha=0.9, colorbar=True, show_traces=True)
brain = stc_nfblab_right.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'NFBLab method right', 'title',
               font_size=14)

vertno_max_idx, time_max = stc_nfblab_right.get_peak(hemi=None,vert_as_index=True)
fig, ax = plt.subplots()
ax.plot(1e3 * stc_nfblab_right.times, stc_nfblab_right.data[vertno_max_idx])
ax.set(xlabel='time (ms)', ylabel='%s value' % method)
pass






# Get protocol with ERP
# make MNE object
# Get events
# pre processing of EEG data
#   filtering
#   ICA removal?
# look at ERP source

# Do above for entire