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
from pynfb.serializers import read_spatial_filter
from pynfb.signal_processing.filters import FFTBandEnvelopeDetector, ExponentialSmoother

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

# ------ Get data files
data_directory = "/Users/christopherturner/Documents/EEG_Data/pilot_202201" # This is the directory where all participants are in

# get participants
participants = next(os.walk(data_directory))[1]

# Get scalp, sham, source data for each participant
experiment_dirs = {}
for p in participants:
# TODO: fix data file structure (when have a solid test setup) - maybe include 'sc', 'so', 'sh' in the data directory names
#       and allocate this way - this way don't have to sort into separate folders. - if do 2 tasks per session, then also don't have to copy
    experiment_dirs[p] = {}
    experiment_dirs[p]["scalp"] = next(os.walk(os.path.join(data_directory, p, "scalp")))[1]
    experiment_dirs[p]["source"] = next(os.walk(os.path.join(data_directory, p, "source")))[1]
    experiment_dirs[p]["sham"] = next(os.walk(os.path.join(data_directory, p, "sham")))[1]

experiment_data = []
for participant, participant_dirs in experiment_dirs.items():
    participant_data = {"participant_id": participant, "session_data": []}
    if participant:# == "kk":
        for session, session_dirs in participant_dirs.items():
            session_data = {}
            for task_dir in session_dirs:
                if "nfb" in task_dir:
                    task_data = {}
                    h5file = os.path.join(data_directory, participant, session, task_dir, "experiment_data.h5")

                    # Put data in pandas data frame
                    df1, fs, channels, p_names = load_data(h5file)
                    df1['sample'] = df1.index

                    # Get baseline only data
                    df1 = df1.loc[df1.block_name == "baseline"]

                    # Drop non eeg data
                    drop_cols = [x for x in df1.columns if x not in channels]
                    drop_cols.extend(['MKIDX', 'EOG', 'ECG'])

                    eeg_data = df1.drop(columns=drop_cols)

                    # Rescale the data (units are microvolts - i.e. x10^-6
                    eeg_data = eeg_data * 1e-6

                    #------- NFB LAB FILTERING
                    #TODO: above but use the NFBLab filtering system on all the data
                    bandpass = (8, 12)
                    smoothing_factor = 0.3
                    smoother = ExponentialSmoother(smoothing_factor)
                    left_alpha_chs = "PO7=1;P5=1;O1=1"
                    channel_labels = eeg_data.columns
                    spatial_matrix = read_spatial_filter(left_alpha_chs, fs, channel_labels=channel_labels)
                    filtered_chunk = np.dot(eeg_data, spatial_matrix)
                    n_samples = len(filtered_chunk)
                    signal_estimator = FFTBandEnvelopeDetector(bandpass, fs, smoother, n_samples)
                    current_chunk = signal_estimator.apply(filtered_chunk)
                    fig = px.line(current_chunk[:500], title=f"{participant}>{session}>{task_dir}")
                    fig.show()

                    #------- MNE OBJECTS
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

                    # Save EEG with stim data as a brainvision file
                    # brainvision_file = os.path.join(os.getcwd(), f'{p}_{session}_{task_dir}-bv.vhdr')
                    # write_raw_brainvision(m_raw, brainvision_file, events=True)

                    # set the reference to average
                    m_raw.set_eeg_reference(projection=True)

                    # plot the psd
                    print(f"-- PLOTTING: {participant}>{session}>{task_dir}")
                    m_raw.plot_psd(spatial_colors=True)
                    m_raw.plot_psd(average=True)
                    m_raw.plot_psd(spatial_colors=True, xscale='log')

                    m_resting = m_raw.copy()
                    m_resting.filter(l_freq=1, h_freq=40)#, iir_params={"order": 2}, method="iir")
                    m_alpha = m_raw.copy()
                    m_alpha.filter(l_freq=8, h_freq=12)#, iir_params={"order": 2}, method="iir")
                    m_resting.plot_psd(spatial_colors=True, xscale='log')
                    m_resting.plot_psd(average=True, xscale='log')
                    m_alpha.plot_psd(spatial_colors=True, xscale='log')
                    m_alpha.plot_psd(average=True, xscale='log')
                    pass

