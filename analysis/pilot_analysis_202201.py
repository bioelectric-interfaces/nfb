"""
This is a script to analyse pilot data for the first experiment

following procedures here:
Müller, J. A., Wendt, D., Kollmeier, B., & Brand, T. (2016). Comparing eye tracking with electrooculography for measuring individual sentence comprehension duration. PLoS ONE, 11(10), 1–22. https://doi.org/10.1371/journal.pone.0164627
"""
import matplotlib.pyplot as plt
import numpy as np
from pynfb.serializers.hdf5 import load_h5py_all_samples, load_h5py_protocol_signals, load_h5py_protocols_raw, load_h5py
from utils.load_results import load_data
import os
import glob
import pandas as pd
import plotly.express as px
from scipy.signal import butter, lfilter, freqz
import mne

import analysis_functions as af

# ------ low pass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

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
    for session, session_dirs in participant_dirs.items():

        # free viewing vars
        session_data = {"session_name": session}
        pre_fb_ratio = 0
        post_fb_ratio = 0
        pre_fb_median = 0
        post_fb_median = 0

        for task_dir in session_dirs:
            h5file = os.path.join(data_directory, participant, session, task_dir, "experiment_data.h5") #"/Users/christopherturner/Documents/EEG Data/ChrisPilot20220110/0-pre_task_ct01_01-10_16-07-00/experiment_data.h5"
            # h5file = "/Users/christopherturner/Documents/EEG Data/ChrisPilot20220110/0-post_task_ct01_01-10_16-55-15/experiment_data.h5"

            # Put data in pandas data frame
            df1, fs, channels, p_names = load_data(h5file)
            df1['sample'] = df1.index

            # Low pass filter (20Hz) the EOG/ECG channels
            cutoff = 20
            df1['ECG_FILTERED'] = butter_lowpass_filter(df1['ECG'], cutoff, fs)
            df1['EOG_FILTERED'] = butter_lowpass_filter(df1['EOG'], cutoff, fs)

            protocol_data = af.get_protocol_data(df1, channels=channels, p_names=p_names)

            # Get free view task stats
            if "pre" in task_dir:
                # get initial fixiation bias
                pre_fb_ratio, pre_fb_median = af.get_task_fixation_bias(protocol_data)
            if "post" in task_dir:
                post_fb_ratio, post_fb_median = af.get_task_fixation_bias(protocol_data)

        # Get change in free viewing fixation bias
        session_data["delta_fb_ratio"] = post_fb_ratio - pre_fb_ratio
        session_data["delta_fb_median"] = post_fb_median - pre_fb_median
        participant_data["session_data"].append(session_data)

        # Do permutation test for individual data i.e. is this change significant <- NOT SURE IF NEEDED?

    experiment_data.append(participant_data)
    # TODO: save this off so don't have to run the entire script again
pass

# Free view analysis
af.free_view_analysis()


pass


#--- NFB trials ----
# Prepare EEG signals
# Get source activation for left and right parietal and occipital for each trial
# plot time course of source activation over all trials

# Look at probe responses

# average AAI over 4 periods
# look at change in AAI over each period
# Look at correct trials over time for participant
# plot score over time for participant
