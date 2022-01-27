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


h5file = "/Users/christopherturner/Documents/EEG_Data/system_testing/raw_data/finger_move_nosound_01-26_18-40-01/experiment_data.h5"

# Put data in pandas data frame
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

#
# Drop non eeg data
drop_cols = [x for x in df1.columns if x not in channels]
drop_cols.extend(['MKIDX', 'EOG', 'ECG'])
eeg_data = df1.drop(columns=drop_cols)
#
# create an MNE info
m_info = mne.create_info(ch_names=list(eeg_data.columns), sfreq=fs, ch_types=['eeg' for ch in list(eeg_data.columns)])
#
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
#
#
# set the reference to average
m_raw.set_eeg_reference(projection=True)
#
# # low pass at 40hz
m_filt = m_raw.copy()
m_filt.filter(l_freq=1, h_freq=40)
mne.viz.plot_raw_psd(m_filt)

# get block transitions
right_block = df1.loc[df1.block_number == 4]
wait_1_block = df1.loc[df1.block_number == 5]
left_block = df1.loc[df1.block_number == 6]
wait_2_block = df1.loc[df1.block_number == 7]
both_block = df1.loc[df1.block_number == 8]

m_filt.crop(tmin=right_block['sample'].iloc[0]/fs, tmax=right_block['sample'].iloc[-1]/fs)



info = m_filt.info
print(info.get('chs')[0]['loc'])

tstep = 1. / info['sfreq']

# DO SOME FILTERING (get beta for finger movements?)
# 'Beta' ~= 13Hz-25hz
fmin = 13
fmax = 25
m_filt.load_data()
raw_twitch = m_filt.filter(fmin, fmax, n_jobs=1,  # use more jobs to speed up.
                   l_trans_bandwidth=1,  # make sure filter params are the same
                   h_trans_bandwidth=1)  # in each band and skip "auto" option.
mne.viz.plot_raw_psd(m_filt)

########################################
# FORWARD SOLUTION
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
subject = 'fsaverage'
# --I think this 'trans' is like the COORDS2TRANSFORMATIONMATRIX
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation #TODO: figure out how to get this for own/MRI data
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif') #TODO: does this just need to be caluculated differently if using own MRI data? - look at mne.setup_source_space (example in the mixed_source_space_inverse.py)
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
fwd = mne.make_forward_solution(m_filt.info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=1)
src=fwd['src']

#--------------------------
#!!!WRITE DATA FOR LORETAKEY
# mne_data2asc(raw, op.join("output", "simulated_data"))
#!!!!!!!!!!!!!!!!!!!!!!!!!!!
#--------------------------


########################################
# SOURCE STUFF
noise_cov = mne.compute_raw_covariance(m_filt, tmax=0.5)  # LOOKS LIKE THIS NEEDS TO BE JUST RAW DATA - i.e. WITH NO EVENTS (OTHERWISE NEED TO DO THE EPOCH ONE AND FIND EVENTS) - PROBABLY GET THIS FROM BASELINE
loose = 0.2
depth = 0.8
label = None  # NOTE!!! LABELS ARE ONLY SUPPORTED WHEN DOING THIS IN SURFACE MODE
source_type = "surface"

# Get the forward solution for SURFACE
# if source_type == "surface":
# #     label_name = surface_labels  # TODO: check this - why do i get signal out with the bci test regardless of the label (though the amplitudes do change a little)
# #     label = sd.get_labels(label_name)[0]
# # elif source_type == "volume":
# #     loose = 'auto'
# #     depth = None

inv = make_inverse_operator(info, fwd, noise_cov, loose=loose, depth=depth)


method = "sLORETA"
snr = 3.
lambda2 = 1. / snr ** 2


label_name = "inferiorparietal-lh"+"superiorparietal-lh"

stc = apply_inverse_raw(m_filt, inv, lambda2, method=method, verbose=True)

vertno_max, time_max = stc.get_peak(hemi=None)

# subjects_dir = data_path + '/subjects'
surfer_kwargs = dict(
    hemi='both', subjects_dir=subjects_dir,
    clim='auto', views='lateral', #clim=dict(kind='value', lims=[8, 12, 15])
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10,
    surface='Pial', transparent=True, alpha=0.9, colorbar=True)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
               font_size=14)

m_filt.plot()
pass