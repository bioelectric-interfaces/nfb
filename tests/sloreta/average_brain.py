import mne
import numpy as np
import pylab as plt

mne.utils.set_config("SUBJECTS_DIR", 'av_brain', set_env=True)

# 1 make fsaverage
#mne.create_default_subject(subjects_dir='av_brain', fs_home=r'D:\soft\freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0\freesurfer')

# 2 compute source space into "bem" folder
# src = mne.setup_source_space('fsaverage', )
src = mne.read_source_spaces(r'C:\Users\nsmetanin\PycharmProjects\nfb\tests\sloreta\av_brain\fsaverage\bem\fsaverage-oct-6-src.fif')


# make bem surf
#model = mne.make_bem_model(subject='fsaverage', ico=4)
#mne.write_bem_surfaces(r'C:\Users\nsmetanin\PycharmProjects\nfb\tests\sloreta\av_brain\fsaverage\bem\fsaverage-bem-surf.fif', model)


# make bem solution
#surfs = mne.read_bem_surfaces(r'C:\Users\nsmetanin\PycharmProjects\nfb\tests\sloreta\av_brain\fsaverage\bem\fsaverage-bem-surf.fif')
#bem = mne.make_bem_solution(surfs)
#mne.write_bem_solution('bem-solution.fif', bem)
bem = mne.read_bem_solution(r'C:\Users\nsmetanin\PycharmProjects\nfb\tests\sloreta\av_brain\fsaverage\bem\bem-solution-flash.fif')

# set trans
trans = r'C:\Users\nsmetanin\PycharmProjects\nfb\tests\sloreta\av_brain\fsaverage\bem\fsaverage-trans.fif'

# create info
fs = 500
channels = ['Fp1','Fp2','F7','F3','Fz','F4','F8','Ft9','Fc5','Fc1','Fc2','Fc6','Ft10','T7','C3','Cz','C4','T8','Tp9',
              'Cp5','Cp1','Cp2','Cp6','Tp10','P7','P3','Pz','P4','P8','O1','Oz','O2']
info = mne.create_info(ch_names=channels, sfreq=fs, montage=mne.channels.read_montage(kind='standard_primed'), ch_types=['eeg' for ch in channels])

# forward solution
#fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, fname='fsaverage-fwd.fif', meg=False, eeg=True, mindist=5.)
fwd = mne.read_forward_solution(r'C:\Users\nsmetanin\PycharmProjects\nfb\tests\sloreta\av_brain\fsaverage\bem\fsaverage-fwd.fif', surf_ori=True)


# data
data = np.random.normal(loc=0, scale=0.00001, size=(5000, len(info["ch_names"])))
raw = mne.io.RawArray(data.T, info)
noise_cov = mne.compute_raw_covariance(raw)

# inverse
inv = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8)


raw.set_eeg_reference()
stc = mne.minimum_norm.apply_inverse_raw(raw, inv, 0.1)
plt.plot(1e3 * stc.times, stc.data[::150, :].T)
plt.show()