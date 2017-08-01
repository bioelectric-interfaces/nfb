from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator

print(__doc__)

data_path = sample.data_path()
fname = data_path
fname += '/MEG/sample/sample_audvis-eeg-oct-6-eeg-inv.fif'

inv = read_inverse_operator(fname)
print(inv.keys())


print("Method: %s" % inv['methods'])
print("fMRI prior: %s" % inv['fmri_prior'])
print("Number of sources: %s" % inv['nsource'])
print("Number of channels: %s" % inv['nchan'])