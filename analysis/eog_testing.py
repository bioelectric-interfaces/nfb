"""
This is a script to check eog eye tracking data for the first experiment

following procedures here:
Müller, J. A., Wendt, D., Kollmeier, B., & Brand, T. (2016). Comparing eye tracking with electrooculography for measuring individual sentence comprehension duration. PLoS ONE, 11(10), 1–22. https://doi.org/10.1371/journal.pone.0164627
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

# ------ Get data files
#
# h5file = "/Users/christopherturner/Documents/EEG Data/eyecalibtest_20221401/horizontal/eye_calib_9pt_01-14_12-41-22/experiment_data.h5" # Horizontal 9 pt calibration
h5file = "/Users/christopherturner/Documents/EEG Data/eyecalibtest_20221401/horizontal/eye_calib_number_01-14_12-32-27/experiment_data.h5" # horizontal numbers
# h5file = "/Users/christopherturner/Documents/EEG Data/eyecalibtest_20221401/horizontal/eye_calib_letter_01-14_12-36-24/experiment_data.h5" # horizontal letters
# h5file = "/Users/christopherturner/Documents/EEG Data/eyecalibtest_20221401/horizontal/eye_calib_9pt_01-14_12-43-17/experiment_data.h5" # horizontal lines
# h5file = "/Users/christopherturner/Documents/EEG Data/eyecalibtest_20221401/horizontal/eye_calib_9pt_01-14_12-46-40/experiment_data.h5" # horizontal shapes


# h5file = "/Users/christopherturner/Documents/EEG Data/eyecalibtest_20221401/horizontal_vertical/eye_calib_9pt_01-14_12-53-04/experiment_data.h5" # horiz_vert 9 pt calibration
# h5file = "/Users/christopherturner/Documents/EEG Data/eyecalibtest_20221401/horizontal_vertical/eye_calib_9pt_01-14_12-55-18/experiment_data.h5" # horiz_vert 9 pt lines
# h5file = "/Users/christopherturner/Documents/EEG Data/eyecalibtest_20221401/horizontal_vertical/eye_calib_9pt_01-14_12-57-04/experiment_data.h5" # horiz_vert 9 pt shapes

# h5file = "/Users/christopherturner/Documents/EEG Data/pilot_202201/ct/scalp/0-post_task_ct01_01-10_16-55-15/experiment_data.h5" #pilotdata

# Put data in pandas data frame
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

# Low pass filter (20Hz) the EOG/ECG channels
cutoff = 20
df1['ECG_FILTERED'] = butter_lowpass_filter(df1['ECG'], cutoff, fs)
df1['EOG_FILTERED'] = butter_lowpass_filter(df1['EOG'], cutoff, fs)

# Extract the individual protocols
protocol_data = {}
block_numbers = df1['block_number'].unique()
protocol_names = [f"{a_}{b_}" for a_, b_ in zip(p_names, block_numbers)]
channels_signal = channels.copy()
channels_signal.append("EOG_FILTERED")
channels_signal.append("ECG_FILTERED")
df2 = pd.melt(df1, id_vars=['block_name', 'block_number', 'sample', 'choice', 'probe', 'answer'],
              value_vars=channels_signal, var_name="channel", value_name='data')

for protocol_n in block_numbers:
    protocol_data[protocol_names[protocol_n - 1]] = df2.loc[df2['block_number'] == protocol_n]

eog_data = protocol_data['eye_calib2'].loc[protocol_data['eye_calib2']['channel'].isin(["ECG_FILTERED", "EOG_FILTERED"])]
# eog_data = protocol_data['EyeCalib2'].loc[protocol_data['EyeCalib2']['channel'].isin(["ECG_FILTERED", "EOG_FILTERED"])]
eog_h = eog_data.loc[eog_data['channel'] == "EOG_FILTERED"]['data'].reset_index(drop=True)
eog_v = eog_data.loc[eog_data['channel'] == "ECG_FILTERED"]['data'].reset_index(drop=True)
eog_spatial = pd.DataFrame({"eog_h": eog_h, "eog_v": eog_v})


# UNCALIBRATED
fig = px.line(eog_data, x="sample", y="data", color='channel')
fig.show()

fig = px.line(eog_spatial, x=eog_spatial.index, y="eog_h")
fig.show()

fig = px.line(eog_spatial, x="eog_h", y="eog_v")
fig.show()

st = 0
end = len(eog_spatial)

# GET SHAPES FROM HORIZONTAL-VERT
# Get the square
# st = 3865
# end = 16103
#
# # uparrow
# st = 18265
# end = 29353
#
# # leftarrow
# st = 32468
# end = 40500

# GET LINES FROM HORIZONTAL-VERT
# left-right
# st = 12153
# end = 17908

# GET SHAPES FROM HORIZONTAL
# Get the square
# st = 2463
# end = 24472
# #
# # # uparrow
# st = 29159
# end = 43038
# #
# # leftarrow
# st = 32468
# end = 40500

x = eog_spatial.iloc[st:end, :]['eog_h']
y = eog_spatial.iloc[st:end, :]['eog_v']

transform = Affine2D().rotate_deg(45)
fig, ax = plt.subplots()

trans_data = transform + ax.transData
im = ax.plot(x,y, transform=trans_data)

fig.suptitle('matplotlib.axis.Axis.set_transform() \
function Example\n', fontweight ="bold")

plt.show()

pass