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
drop_cols.extend(['MKIDX', 'EOG', 'ECG'])
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
m_raw.set_eeg_reference(projection=True)

#########################
# PRE-PROCESSING
#########################
"""
Gets rid of artefacts (various different sources of noise)
Can be done automatically but should really be done visually to ensure good results
"""

#-----------------------
# TEMPORAL FILTERING
#-----------------------
"""
Get a specific frequency band and remove non-relavant frequencies
"""
#non-causal, Infinite Impulse Response (IIR) Butterworth filter of 2nd order
m_resting = m_raw.filter(l_freq=1, h_freq=40, iir_params={"order": 2}, method="iir")
m_alpha = m_raw.filter(l_freq=8, h_freq=12, iir_params={"order": 2}, method="iir")

#-----------------------
# BAD ELECTRODE DETECTION AND INTERPOLATION
#-----------------------
"""
Remove electrodes whose values go above a certain range (from the resting filtered data)
mne tutorial: https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html
"""

#-----------------------
# ICA
#-----------------------
"""
do ica correction to remove artefacts
do this on the baseline data and apply to the entire raw dataset
"""


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


#-----------------------
# DOWN SAMPLING
#-----------------------
"""
Down sample the data to 4* highest frequency
Do this AFTER epoching (as it jitters the timings)
"""

