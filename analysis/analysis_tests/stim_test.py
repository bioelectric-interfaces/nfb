"""
Script to look at the difference in offsets between brain vision markers and lsl markers
"""
import mne
from utils.load_results import load_data
import numpy as np
import pandas as pd
import plotly_express as px
import csv

# get the brain vision stuff
# fname_raw = "/Users/2354158T/Documents/EEG_Data/bv_stim_test/posner_marker_test2.vhdr"
fname_raw = "/Users/christopherturner/Documents/EEG_Data/testing_20220614/brainvision_posner/posner_testing_20221614.vhdr"
raw = mne.io.read_raw_brainvision(fname_raw)
events_from_annot, event_dict = mne.events_from_annotations(raw)
# split the data based on annotations
epochs = mne.Epochs(raw, events_from_annot)

# Get the lengths of all the epochs
epochs.get_data()
# epochs['3'][1].plot()

# Get the nfblab lsl events
# h5file = "/Users/2354158T/Documents/EEG_Data/bv_stim_test/posner_test_markers_bv_06-10_13-19-09/experiment_data.h5"
h5file = "/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-posner_task_test_psychopy_06-14_16-55-03/experiment_data.h5"
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

x = len(events_from_annot)-1
diff = []
while x != 0:
    diff.insert(0, events_from_annot[x][0] - events_from_annot[x-1][0])
    x = x-1

nfb_events = df1[df1.EVENTS>0]
x_nfb = len(nfb_events) -1
diff_nfb = []
while x_nfb != 0:
    diff_nfb.insert(0, nfb_events['sample'].iloc[x_nfb] - nfb_events['sample'].iloc[x_nfb-1])
    x_nfb = x_nfb-1

diff.pop(0) # remove extra beginning event in BV in NFBLAB
diff_nfb.pop(0) # remove extras at beginning and end in BV
diff_nfb.pop(-1)
subtracted = np.subtract(diff, diff_nfb) #[element1 - element2 for (element1, element2) in zip(diff, diff_nfb)]
px.violin(subtracted, box=True, points='all',
          title=f"block:").show()  # , range_y=[200, 800]).show()

# read in the psychopy results
pp_df = pd.DataFrame(dict(marker=[], time=[]))
# psychopy_results = pd.read_csv('/Users/2354158T/Documents/EEG_Data/bv_stim_test/2_posner_2022-06-10_13h18.45.421.csv', squeeze=True)
psychopy_results = pd.read_csv('/Users/christopherturner/Documents/EEG_Data/testing_20220614/posner/1_posner_2022-06-14_16h54.47.522.csv', squeeze=True)
for index, row in psychopy_results.iterrows():
    pp_df = pp_df.append({'marker':row['cue'], 'time': row['fc.started']}, ignore_index = True)
    pp_df = pp_df.append({'marker':3, 'time': row['fixation_cross.started']}, ignore_index = True)

# Drop extra rows in beginning
pp_df = pp_df.drop([pp_df.index[0] , pp_df.index[1]])

x_pp = len(pp_df)-1
diff_pp = []
while x_pp != 0:
    diff_pp.insert(0, round((pp_df['time'].iloc[x_pp] - pp_df['time'].iloc[x_pp-1])*1000)) # approx samples
    x_pp = x_pp-1


subtracted_parallel_pp = np.subtract(diff, diff_pp) # This is the difference between the pyschopy timestamped blocks and the brainvision recorded events. It is the same (+-1 is rounding error)


print("done")