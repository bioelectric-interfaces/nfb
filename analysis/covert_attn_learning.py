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
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw, apply_inverse, apply_inverse_epochs

from pynfb.helpers import roi_spatial_filter as rsf
from philistine.mne import savgol_iaf

if platform.system() == "Windows":
    userdir = "2354158T"
else:
    userdir = "christopherturner"

task_data = {}

task_data[
    "s1"] = f"/Users/{userdir}/Documents/EEG_Data/system_testing/ksenia_cvsa/cvsa_02-05_15-39-15/experiment_data.h5"  # Ksenia cvsa tasks 1
task_data[
    "s2"] = f"/Users/{userdir}/Documents/EEG_Data/system_testing/ksenia_cvsa/cvsa_02-05_15-47-03/experiment_data.h5"  # Ksenia cvsa tasks 2

task_data[
    "s3"] = f"/Users/{userdir}/Documents/EEG_Data/system_testing/ksenia_cvsa/cvsa_02-15_15-21-37/experiment_data.h5"  # Ksenia cvsa 3 **
task_data[
    "s4"] = f"/Users/{userdir}/Documents/EEG_Data/system_testing/ksenia_cvsa/cvsa_02-15_15-37-24/experiment_data.h5"  # Ksenia cvsa 4

for session, data_file in task_data.items():
    session_data = {}
    # Put data in pandas data frame
    df1, fs, channels, p_names = load_data(data_file)
    df1['sample'] = df1.index

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
    m_info = mne.create_info(ch_names=list(eeg_data.columns), sfreq=fs,
                             ch_types=['eeg' for ch in list(eeg_data.columns)])

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

    # TODO: ICA for each

    # Filter in alpha band
    m_filt = m_raw.copy()
    m_filt.filter(8, 14, n_jobs=1,  # use more jobs to speed up.
                  l_trans_bandwidth=1,  # make sure filter params are the same
                  h_trans_bandwidth=1)  # in each band and skip "auto" option.

    events = mne.find_events(m_raw, stim_channel='STI')
    reject_criteria = dict(eeg=100e-6)

    left_chs = ['PO7=1']
    right_chs = ['PO8=1']

    # - - DO baseline epochs
    # fig_bl_eo1, aai_baseline_eo1, bl_dataframe_eo1, bl_epochs_eo1 = af.do_baseline_epochs(df1, m_filt, left_chs,
    #                                                                                       right_chs, fig=None,
    #                                                                                       fb_type="active",
    #                                                                                       baseline_name='bl_eo1',
    #                                                                                       block_number=2)
    # fig_bl_ec1, aai_baseline_ec1, bl_dataframe_ec1, bl_epochs_ec1 = af.do_baseline_epochs(df1, m_filt, left_chs,
    #                                                                                       right_chs, fig=None,
    #                                                                                       fb_type="active",
    #                                                                                       baseline_name='bl_ec1',
    #                                                                                       block_number=4)
    # fig_bl_eo2, aai_baseline_eo2, bl_dataframe_eo2, bl_epochs_eo2 = af.do_baseline_epochs(df1, m_filt, left_chs,
    #                                                                                       right_chs, fig=None,
    #                                                                                       fb_type="active",
    #                                                                                       baseline_name='bl_eo2',
    #                                                                                       block_number=83)
    # fig_bl_ec2, aai_baseline_ec2, bl_dataframe_ec2, bl_epochs_ec2 = af.do_baseline_epochs(df1, m_filt, left_chs,
    #                                                                                       right_chs, fig=None,
    #                                                                                       fb_type="active",
    #                                                                                       baseline_name='bl_ec2',
    #                                                                                       block_number=85)
    # get task epochs
    epochs = mne.Epochs(m_filt, events, event_id=event_dict, tmin=0, tmax=7, baseline=None,
                        preload=True, detrend=1)

    # Get AAI (using NFB method) on epochs
    e_mean1, e_std1, epoch_pwr1 = af.get_nfb_epoch_power_stats(epochs['left_probe'], fband=(8, 14), fs=1000,
                                                               channel_labels=epochs.info.ch_names, chs=["PO7=1"], fft_samps=1000)
    e_mean2, e_std2, epoch_pwr2 = af.get_nfb_epoch_power_stats(epochs['left_probe'], fband=(8, 14), fs=1000,
                                                               channel_labels=epochs.info.ch_names, chs=["PO8=1"], fft_samps=1000)
    aai_nfb_left = (epoch_pwr1 - epoch_pwr2) / (epoch_pwr1 + epoch_pwr2)

    dataframes_aai_cue = []
    dataframes_aai = []
    aai_nfb_left = (epoch_pwr1 - epoch_pwr2) / (epoch_pwr1 + epoch_pwr2)

    aai_bl = 0
    aai_threshold = aai_bl + 0.2
    epoch_rat_abv_th = []
    aai_total = []
    for ep in aai_nfb_left:
        epoch_rat_abv_th.append(np.count_nonzero(ep > aai_threshold) / ep.size)
        aai_total = aai_total + ep.tolist()[0]

    session_data['epoch_rat_abv_th_left'] = epoch_rat_abv_th
    session_data['total_med_aai_left'] = aai_total

    pass
    # TODO:
    #   get median of AAI for each quarter over all sessions (Q1, 2, 3, 4, 5, 6, 7...)
    #   Get median AAI over all sessions
