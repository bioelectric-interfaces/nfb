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
data_directory = "/Users/2354158T/Documents/EEG_Data/pilot_202201" # This is the directory where all participants are in

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
                    # Get all samples where choice protocol starts
                    # Put the transition (event) data back in the original dataframe
                    df1['protocol_change'] = df1['block_number'].diff()
                    df1['choice_events'] = df1.apply(lambda row: row.protocol_change if row.block_name == "Input" else 0, axis = 1)

                    # Create the events list for the protocol transitions
                    probe_events = df1[['choice_events']].to_numpy()
                    event_type = 1
                    event_dict = {'choice_transition': event_type}

                    # Drop non eeg data
                    eeg_data = df1.drop(
                        columns=['signal_Alpha_Left', 'signal_Alpha_Right', 'signal_AAI', 'events', 'reward', 'choice', 'answer', 'probe', 'block_name',
                                 'block_number', 'sample', 'MKIDX', 'protocol_change', 'choice_events'])


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

                    # Create the stim channel
                    info = mne.create_info(['STI'], m_raw.info['sfreq'], ['stim'])
                    stim_raw = mne.io.RawArray(probe_events.T, info)
                    m_raw.add_channels([stim_raw], force_update_info=True)

                    # Save EEG with stim data as a brainvision file
                    # brainvision_file = os.path.join(os.getcwd(), f'{p}_{session}_{task_dir}-bv.vhdr')
                    # write_raw_brainvision(m_raw, brainvision_file, events=True)

                    # set the reference to average
                    m_raw.set_eeg_reference(projection=True)

                    # # low pass at 40hz
                    m_filt = m_raw.copy()
                    m_filt.filter(l_freq=0, h_freq=40)

                    # # epoch the data
                    events = mne.find_events(m_raw, stim_channel='STI')
                    picks = [c for c in m_info.ch_names if c not in ['EOG', 'ECG']]
                    epochs = mne.Epochs(m_filt, events, event_id=event_dict, tmin=-0.2, tmax=0.5,
                                        preload=True, picks=picks, detrend=1)
                    # fig = epochs.plot(events=events, scalings={"eeg":10})

                    # average epochs
                    choice_transition = epochs['choice_transition'].average()
                    fig2 = choice_transition.plot(spatial_colors=True)
                    # choice_transition.plot_joint()

                    # plot topomap
                    choice_transition.plot_topomap(times=[-0.2, 0.1, 0.4], average=0.05)

                    pass
                    # ---------- SOURCE RECONSTRUCTION
                    # TODO: do this for epoched data (once get the above working)
                    # - first do this over the transition to choice
                    # TODO: make sure this is done the same way as the GUI (or compare the two if this one works)
                    # do initial calcs
                    # info = m_filt.info
                    # noise_cov = mne.compute_raw_covariance(m_filt, tmax=0.5) # LOOKS LIKE THIS NEEDS TO BE JUST RAW DATA - i.e. WITH NO EVENTS (OTHERWISE NEED TO DO THE EPOCH ONE AND FIND EVENTS) - PROBABLY GET THIS FROM BASELINE
                    # loose = 0.2
                    # depth = 0.8
                    # label = None # NOTE!!! LABELS ARE ONLY SUPPORTED WHEN DOING THIS IN SURFACE MODE
                    #
                    # # Get the forward solution for the specified source localisation type
                    # fs_dir = fetch_fsaverage(verbose=True)
                    # # --I think this 'trans' is like the COORDS2TRANSFORMATIONMATRIX
                    # trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
                    # src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
                    # bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
                    # fwd = mne.make_forward_solution(info, trans=trans, src=src,
                    #                                 bem=bem, eeg=True, mindist=5.0, n_jobs=1)
                    #
                    # # label_name = surface_labels # TODO: check this - why do i get signal out with the bci test regardless of the label (though the amplitudes do change a little)
                    # # label = sd.get_labels(label_name)[0]
                    #
                    # inv = make_inverse_operator(info, fwd, noise_cov, loose=loose, depth=depth)
                    # # TODO: do this for epochs
                    #
                    # snr = 1.0  # use smaller SNR for raw data
                    # lambda2 = 1.0 / snr ** 2
                    # method = "sLORETA"  # use sLORETA method (could also be MNE or dSPM)
                    # start, stop = m_raw.time_as_index([0, 2])
                    # stc = apply_inverse_raw(m_filt, inv, lambda2, method, label,
                    #                         start, stop, pick_ori=None)
                    #
                    # # Save result in stc files
                    # # stc.save('mne_%s_raw_inverse_%s' % (method, label_name))
                    #
                    #
                    # vertno_max, time_max = stc.get_peak(hemi=None)
                    #
                    # surfer_kwargs = dict(
                    #     hemi='both',
                    #     clim='auto', views='lateral',  # clim=dict(kind='value', lims=[8, 12, 15])
                    #     initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10,
                    #     surface='Pial', transparent=True, alpha=0.9, colorbar=True)
                    # brain = stc.plot(**surfer_kwargs)
                    # brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
                    #                scale_factor=0.6, alpha=0.5)
                    # brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
                    #                font_size=14)
                    #
                    # # Do a matplotlib to hold the brain plot in place!!
                    # plt.plot(1e3 * stc.times, stc.data[::100, :].T)
                    # plt.xlabel('time (ms)')
                    # plt.ylabel('%s value' % method)
                    # plt.show()
                    pass

# Get protocol with ERP
# make MNE object
# Get events
# pre processing of EEG data
#   filtering
#   ICA removal?
# look at ERP source

# Do above for entire