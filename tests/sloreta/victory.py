import mne
import numpy as np
import pylab as plt
from mne.datasets import sample

# create data
fs = 500
ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','Ft9','Fc5','Fc1','Fc2','Fc6','Ft10','T7','C3','Cz','C4','T8','Tp9',
              'Cp5','Cp1','Cp2','Cp6','Tp10','P7','P3','Pz','P4','P8','O1','Oz','O2']
info = mne.create_info(ch_names=ch_names, sfreq=fs, montage=mne.channels.read_montage(kind='standard_1005'), ch_types=['eeg' for ch in ch_names])
data = np.random.normal(loc=0, scale=0.00001, size=(5000, len(info["ch_names"])))
raw = mne.io.RawArray(data.T, info)
#raw.plot()
#plt.show()

# create cov matrix
noise_cov = mne.compute_raw_covariance(raw)
mne.viz.plot_cov(noise_cov, raw.info)

# forward solution
fname_fwd = sample.data_path() + '/MEG/sample/sample_audvis-meg-oct-6-fwd.fif'
fwd = mne.read_forward_solution(fname_fwd, surf_ori=True)

