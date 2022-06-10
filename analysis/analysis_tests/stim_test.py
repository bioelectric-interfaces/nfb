
import mne
from utils.load_results import load_data
import numpy as np

# get the brain vision stuff
fname_raw = "/Users/2354158T/Documents/EEG_Data/bv_stim_test/posner_marker_test2.vhdr"
raw = mne.io.read_raw_brainvision(fname_raw)
events_from_annot, event_dict = mne.events_from_annotations(raw)
# split the data based on annotations
epochs = mne.Epochs(raw, events_from_annot)

# Get the lengths of all the epochs
epochs.get_data()
# epochs['3'][1].plot()

# Get the nfblab lsl events
h5file = "/Users/2354158T/Documents/EEG_Data/bv_stim_test/posner_test_markers_bv_06-10_13-19-09/experiment_data.h5"
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
subtracted = np.subtract(diff, diff_nfb) #[element1 - element2 for (element1, element2) in zip(diff, diff_nfb)]

print("done")