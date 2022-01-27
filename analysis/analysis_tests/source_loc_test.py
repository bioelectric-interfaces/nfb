"""
File to analyse the ERP data from the probe
this does:
    A) look at ERP power in left and right side
    B) look at source location for left and right ERPs
"""
import sys
import os

from philistine.mne import write_raw_brainvision

sys.path.append(f"{os.getcwd()}")

import matplotlib.pyplot as plt
import numpy as np
from mne.channels import make_standard_montage
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw

from utils.load_results import load_data
import glob
import pandas as pd
import plotly.express as px
from scipy.signal import butter, lfilter, freqz
import mne

mne.viz.set_3d_backend('pyvista')

# ------ Get data files
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
#     if participant == "sh":
#         for session, session_dirs in participant_dirs.items():
#             session_data = {}
#             for task_dir in session_dirs:
#                 if "nfb" in task_dir:
task_data = {}
h5file = "/Users/christopherturner/Documents/EEG_Data/system_testing/raw_data/audio_3hz_01-26_18-35-09/experiment_data.h5"

# Put data in pandas data frame
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

#
# Drop non eeg data
drop_cols = [x for x in df1.columns if x not in channels]
drop_cols.extend(['MKIDX'])
eeg_data = df1.drop(columns=drop_cols)
#
# create an MNE info
m_info = mne.create_info(ch_names=list(eeg_data.columns), sfreq=fs, ch_types=['eeg' for ch in list(eeg_data.columns)])
#
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
#
#
# set the reference to average
m_raw.set_eeg_reference(projection=True)
#
# # low pass at 40hz
m_filt = m_raw.copy()
m_filt.filter(l_freq=1, h_freq=40)
#

# get block transitions
right_block = df1.loc[df1.block_number == 4]
wait_1_block = df1.loc[df1.block_number == 5]
left_block = df1.loc[df1.block_number == 6]
wait_2_block = df1.loc[df1.block_number == 7]
both_block = df1.loc[df1.block_number == 8]

m_filt.crop(tmin=right_block['sample'].iloc[0]/fs, tmax=right_block['sample'].iloc[-1]/fs)

# ---------- SOURCE RECONSTRUCTION
# TODO: do this for epoched data (once get the above working)
# - first do this over the transition to choice
# TODO: make sure this is done the same way as the GUI (or compare the two if this one works)
# do initial calcs
info = m_filt.info
noise_cov = mne.compute_raw_covariance(m_filt, tmax=0.5) # LOOKS LIKE THIS NEEDS TO BE JUST RAW DATA - i.e. WITH NO EVENTS (OTHERWISE NEED TO DO THE EPOCH ONE AND FIND EVENTS) - PROBABLY GET THIS FROM BASELINE
loose = 0.2
depth = 0.8
label = None # NOTE!!! LABELS ARE ONLY SUPPORTED WHEN DOING THIS IN SURFACE MODE

# Get the forward solution for the specified source localisation type
fs_dir = fetch_fsaverage(verbose=True)
# --I think this 'trans' is like the COORDS2TRANSFORMATIONMATRIX
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
fwd = mne.make_forward_solution(info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=1)

# label_name = surface_labels # TODO: check this - why do i get signal out with the bci test regardless of the label (though the amplitudes do change a little)
# label = sd.get_labels(label_name)[0]

inv = make_inverse_operator(info, fwd, noise_cov, loose=loose, depth=depth)
# TODO: do this for epochs

snr = 1.0  # use smaller SNR for raw data
lambda2 = 1.0 / snr ** 2
method = "sLORETA"  # use sLORETA method (could also be MNE or dSPM)
start, stop = m_raw.time_as_index([0, 2])
stc = apply_inverse_raw(m_filt, inv, lambda2, method, label,
                        start, stop, pick_ori=None)

# Save result in stc files
# stc.save('mne_%s_raw_inverse_%s' % (method, label_name))


vertno_max, time_max = stc.get_peak(hemi=None)

surfer_kwargs = dict(
    hemi='both',
    clim='auto', views='lateral',  # clim=dict(kind='value', lims=[8, 12, 15])
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10,
    surface='Pial', transparent=True, alpha=0.9, colorbar=True)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
               font_size=14)

# Do a matplotlib to hold the brain plot in place!!
plt.plot(1e3 * stc.times, stc.data[::100, :].T)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
plt.show()
pass

# Get protocol with ERP
# make MNE object
# Get events
# pre processing of EEG data
#   filtering
#   ICA removal?
# look at ERP source

# Do above for entire