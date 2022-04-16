"""
Script to get the screen witdh from the eye calibration
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.axis import Axis
from matplotlib.transforms import Affine2D

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

# -------------
h5file = "/Users/christopherturner/Documents/EEG_Data/eyecalibtest_20221401/horizontal/eye_calib_9pt_01-14_12-41-22/experiment_data.h5" # Horizontal 9 pt calibration

df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

cutoff = 10
df1['ECG_FILTERED'] = butter_lowpass_filter(df1['ECG'], cutoff, fs)
df1['EOG_FILTERED'] = butter_lowpass_filter(df1['EOG'], cutoff, fs)

# Get the left mid side (idx=13) and right mid side (idx=15)
eye_calib_data = df1[df1['block_name'] == 'eye_calib']
left_calib_data = eye_calib_data[eye_calib_data['probe'] == 13]
centre_calib_data = eye_calib_data[eye_calib_data['probe'] == 14]
right_calib_data = eye_calib_data[eye_calib_data['probe'] == 15]

left_calib_mean = (left_calib_data['EOG_FILTERED'] - left_calib_data['ECG_FILTERED']).mean()
centre_calib_mean = (centre_calib_data['EOG_FILTERED'] - centre_calib_data['ECG_FILTERED']).mean()
right_calib_mean = (right_calib_data['EOG_FILTERED'] - right_calib_data['ECG_FILTERED']).mean()

eye_centre = centre_calib_mean
eye_range = left_calib_mean - right_calib_mean

print(f"EYE CENTRE: {eye_centre}, EYE RANGE: {eye_range}")