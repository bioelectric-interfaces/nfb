"""
File to analyse the ERP data from the probe
this does:
    A) look at ERP power in left and right side
    B) look at source location for left and right ERPs
"""
import matplotlib.pyplot as plt
import numpy as np
from mne.channels import make_standard_montage
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw

from pynfb.serializers.hdf5 import load_h5py_all_samples, load_h5py_protocol_signals, load_h5py_protocols_raw, load_h5py
from utils.load_results import load_data
import os
import glob
import pandas as pd
import plotly.express as px
from scipy.signal import butter, lfilter, freqz
import mne

import analysis_functions as af

# ------ Get data files
data_directory = "/Users/christopherturner/Documents/EEG Data/pilot_202201" # This is the directory where all participants are in

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
    if participant == "sh":
        for session, session_dirs in participant_dirs.items():
            session_data = {}
            for task_dir in session_dirs:
                if "nfb" in task_dir:
                    task_data = {}
                    h5file = os.path.join(data_directory, participant, session, task_dir, "experiment_data.h5")

                    # Put data in pandas data frame
                    df1, fs, channels, p_names = load_data(h5file)
                    df1['sample'] = df1.index

                    # get the protocol data and average the AAI for each protocol
                    # protocol_data = af.get_protocol_data(df1, channels=channels, p_names=p_names, eog_filt=False)

                    # probe protocols = 'delay<n>' # TODO: make this work for all protocols
                    pass

                    #-------
                    # look at first delay protocol (protocol 9)
                    # delay_1 = df1.loc[df1['block_number'] == 9].reset_index(drop=True)
                    # # get samples where event happens of probe (1 = right, 2 = left)
                    # probe_samples = delay_1.loc[delay_1['probe'].isin([1, 2])].index.tolist()
                    # event_type = delay_1["probe"].iloc[probe_samples[0]]
                    # event_dict = {'left_probe': 2}
                    # # np.c_[delay_1.index.to_numpy(), np.zeros(len(probe_events)), np.zeros(len(probe_events))]
                    # # probe_events.itemset((probe_samples[0], 2), event_type)
                    # probe_events = np.array([[probe_samples[19], 0, event_type]])
                    # probe_events = probe_events.astype(int)

                    # ------
                    # look at the transition to the first choice screen (when the gabor grating is displayed
                    choice_1 = df1.loc[df1['block_number'] == 10].reset_index(drop=True)
                    # get samples where event happens of probe (1 = right, 2 = left)
                    initial_input_sample = choice_1['sample'][0]
                    # get data 1 second before and after the transition to choice block
                    choice_1_transition = df1.iloc[df1[df1['sample'] == initial_input_sample - 5000].index[0]:df1[df1['sample'] == initial_input_sample + 5000].index[0]].reset_index(drop=True)
                    event_type = 1
                    event_dict = {'choice_transition': event_type}
                    probe_events = np.array([[1000, 0, event_type]])
                    probe_events = probe_events.astype(int)
                    # Put in mne object
                    m_info = mne.create_info(channels, fs, ch_types='eeg', verbose=None)
                    channel_data = choice_1_transition.drop(
                        columns=['signal_Alpha_Left', 'signal_Alpha_Right', 'signal_AAI', 'events', 'reward', 'choice', 'answer', 'probe', 'block_name',
                                 'block_number', 'sample'])
                    m_raw = mne.io.RawArray(channel_data.T, m_info, first_samp=0, copy='auto', verbose=None)
                    # Set the montage
                    montage = make_standard_montage('standard_1020')
                    m_raw.set_montage(montage,on_missing='ignore')

                    # set the reference to average
                    m_raw.set_eeg_reference()

                    # low pass at 40hz
                    m_filt = m_raw.copy()
                    m_filt.filter(l_freq=0.1, h_freq=40)
                    # m_raw.plot(scalings={"eeg":10})

                    # epoch the data
                    epochs = mne.Epochs(m_filt, probe_events, event_id=event_dict, tmin=-0.5, tmax=0.5,
                                        preload=True)
                    fig = epochs.plot(events=probe_events, scalings={"eeg":10})

                    # average epochs # TODO: do this over all data?
                    choice_transition = epochs['choice_transition'].average()
                    fig2 = choice_transition.plot(spatial_colors=True)
                    choice_transition.plot_joint()


                    # ---------- SOURCE RECONSTRUCTION
                    # - first do this over the transition to choice
                    # TODO: make sure this is done the same way as the GUI (or compare the two if this one works)
                    # do initial calcs
                    info = m_raw.info
                    noise_cov = mne.compute_raw_covariance(m_raw, tmax=0.5) # LOOKS LIKE THIS NEEDS TO BE JUST RAW DATA - i.e. WITH NO EVENTS (OTHERWISE NEED TO DO THE EPOCH ONE AND FIND EVENTS) - PROBABLY GET THIS FROM BASELINE
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
                    stc = apply_inverse_raw(m_raw, inv, lambda2, method, label,
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
                    pass

# Get protocol with ERP
# make MNE object
# Get events
# pre processing of EEG data
#   filtering
#   ICA removal?
# look at ERP source

# Do above for entire