from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, apply_inverse_raw
import mne
import pylab as plt
import numpy as np
print(__doc__)

data_path = sample.data_path()
fname = data_path
fname += '/MEG/sample/sample_audvis-eeg-oct-6-eeg-inv.fif'

inv = read_inverse_operator(fname)


fs = 500
channels = ['EEG 001', 'EEG 002', 'EEG 003', 'EEG 004', 'EEG 005', 'EEG 006', 'EEG 007', 'EEG 008', 'EEG 009', 'EEG 010', 'EEG 011', 'EEG 012', 'EEG 013', 'EEG 014', 'EEG 015', 'EEG 016', 'EEG 017', 'EEG 018', 'EEG 019', 'EEG 020', 'EEG 021', 'EEG 022', 'EEG 023', 'EEG 024', 'EEG 025', 'EEG 026', 'EEG 027', 'EEG 028', 'EEG 029', 'EEG 030', 'EEG 031', 'EEG 032', 'EEG 033', 'EEG 034', 'EEG 035', 'EEG 036', 'EEG 037', 'EEG 038', 'EEG 039', 'EEG 040', 'EEG 041', 'EEG 042', 'EEG 043', 'EEG 044', 'EEG 045', 'EEG 046', 'EEG 047', 'EEG 048', 'EEG 049', 'EEG 050', 'EEG 051', 'EEG 052', 'EEG 054', 'EEG 055', 'EEG 056', 'EEG 057', 'EEG 058', 'EEG 059', 'EEG 060']
#info = mne.create_info(ch_names=channels, sfreq=fs, montage=mne.channels.read_montage(kind='standard_primed'), ch_types=['eeg' for ch in channels])
info = inv['info']
#info['sfreq'] = 500
data = np.random.normal(loc=0, scale=0.00001, size=(5000, len(info["ch_names"])))
info.plot_sensors()
raw = mne.io.RawArray(data.T, info)
info.plot_sensors()
#raw.set_eeg_reference()
#raw.plot()
#plt.show()

sources = apply_inverse_raw(raw, inv, 0.01)



print("Method: %s" % inv['methods'])
print("fMRI prior: %s" % inv['fmri_prior'])
print("Number of sources: %s" % inv['nsource'])
print("Number of channels: %s" % inv['nchan'])