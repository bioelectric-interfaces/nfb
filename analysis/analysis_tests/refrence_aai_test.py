"""
script to check if there is a difference in calculating AAI from raw EEG compared to average referenced data
"""

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

h5file = "/Users/christopherturner/Documents/EEG_Data/pilot_202201/ct02/scalp/0-nfb_task_ct02_01-26_16-33-42/experiment_data.h5"
# h5file = "/Users/christopherturner/Documents/EEG_Data/pilot_202201/kk/scalp/0-nfb_task_kk_01-27_18-34-12/experiment_data.h5"
# Put data in pandas data frame
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

channels.append("signal_AAI")

# Get the MNE data
# Drop non eeg data
drop_cols = [x for x in df1.columns if x not in channels]
drop_cols.extend(['MKIDX', 'EOG', 'ECG', 'signal_AAI'])
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
raw_ref_avg = m_raw.copy().set_eeg_reference(ref_channels='average')

# Set a different average
raw_ref_cz = m_raw.copy().set_eeg_reference(ref_channels=['CZ'])

# TODO: THE ACTUAL RAW DATA MIGHT BE AVERAGE REFERENCED ANYWAY - JUST CHECK THE AAI FOR THE ABOVE 2 DIFFERENT TYPES OF REFS


#-----------
# plot first 20s of raw PO7 and PO8
po7 = eeg_data['PO7'][0:2000]
po8 = eeg_data['PO8'][0:2000]
sample = eeg_data['PO7'][0:2000].index
fig = go.Figure()
fig.add_trace(go.Scatter(x=sample, y=po7,
                    mode='lines',
                    name='PO7_raw'))
fig.add_trace(go.Scatter(x=sample, y=po8,
                    mode='lines',
                    name='PO8_raw'))

# plot first 20s of avg ref PO7 and PO7
po7_avg = raw_ref_avg.get_data(picks=['PO7'], start=0, stop=2000)
po7_avg = pd.Series(po7_avg[0])
po8_avg = raw_ref_avg.get_data(picks=['PO8'], start=0, stop=2000)
po8_avg = pd.Series(po8_avg[0])

fig.add_trace(go.Scatter(y=po7_avg,
                    mode='lines',
                    name='PO7_avg'))
fig.add_trace(go.Scatter(y=po8_avg,
                    mode='lines',
                    name='PO8_avg'))

# Plot the first 20s of CZ ref for PO7 and PO8
po7_cz = raw_ref_cz.get_data(picks=['PO7'], start=0, stop=2000)
po7_cz = pd.Series(po7_cz[0])
po8_cz = raw_ref_cz.get_data(picks=['PO8'], start=0, stop=2000)
po8_cz = pd.Series(po8_cz[0])

fig.add_trace(go.Scatter(y=po7_cz,
                    mode='lines',
                    name='PO7_cz'))
fig.add_trace(go.Scatter(y=po8_cz,
                    mode='lines',
                    name='PO8_cz'))
fig.show()


# Plot AAI calculated from raw
aai_duration_samps = 10000
mean_raw_l, std1_raw_l, pwr_raw_l = af.get_nfblab_power_stats_pandas(eeg_data[0:aai_duration_samps], fband=(8, 14), fs=1000,
                                                             channel_labels=m_raw.info.ch_names, chs=["PO7=1"],
                                                             fft_samps=1000)

mean_raw_r, std1_raw_r, pwr_raw_r = af.get_nfblab_power_stats_pandas(eeg_data[0:aai_duration_samps], fband=(8, 14), fs=1000,
                                                             channel_labels=m_raw.info.ch_names, chs=["PO8=1"],
                                                             fft_samps=1000)
aai_raw_left = (pwr_raw_l - pwr_raw_r) / (pwr_raw_l + pwr_raw_r)
fig_aai = go.Figure()
fig_aai.add_trace(go.Scatter(y=aai_raw_left,
                    mode='lines',
                    name='AAI_raw_left'))

# Plot AAI calculated from avg ref data
raw_ref_avg_df = pd.DataFrame(raw_ref_avg.get_data(stop=aai_duration_samps).T, columns=m_raw.info.ch_names)
mean_avg_l, std_avg_l, pwr_avg_l = af.get_nfblab_power_stats_pandas(raw_ref_avg_df, fband=(8, 14), fs=1000,
                                                             channel_labels=m_raw.info.ch_names, chs=["PO7=1"],
                                                             fft_samps=1000)
mean_avg_r, std_avg_r, pwr_avg_r = af.get_nfblab_power_stats_pandas(raw_ref_avg_df, fband=(8, 14), fs=1000,
                                                             channel_labels=m_raw.info.ch_names, chs=["PO8=1"],
                                                             fft_samps=1000)
aai_avg_left = (pwr_avg_l - pwr_avg_r) / (pwr_avg_l + pwr_avg_r)

fig_aai.add_trace(go.Scatter(y=aai_avg_left,
                    mode='lines',
                    name='AAI_avg_left'))

# Plot AAI calculated from CZ ref data
raw_ref_cz_df = pd.DataFrame(raw_ref_cz.get_data(stop=aai_duration_samps).T, columns=m_raw.info.ch_names)
mean_cz_l, std_cz_l, pwr_cz_l = af.get_nfblab_power_stats_pandas(raw_ref_cz_df, fband=(8, 14), fs=1000,
                                                             channel_labels=m_raw.info.ch_names, chs=["PO7=1"],
                                                             fft_samps=1000)
mean_cz_r, std_cz_r, pwr_cz_r = af.get_nfblab_power_stats_pandas(raw_ref_cz_df, fband=(8, 14), fs=1000,
                                                             channel_labels=m_raw.info.ch_names, chs=["PO8=1"],
                                                             fft_samps=1000)
aai_cz_left = (pwr_cz_l - pwr_cz_r) / (pwr_cz_l + pwr_cz_r)

fig_aai.add_trace(go.Scatter(y=aai_cz_left,
                    mode='lines',
                    name='AAI_cz_left'))
fig_aai.show()