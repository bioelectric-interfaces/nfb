import mne
import numpy as np
import pylab as plt

from pynfb.protocols.ssd.topomap_selector_ica import ICADialog

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


# data
real = False
if not real:
    channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9', 'Fc5', 'Fc1', 'Fc2', 'Fc6', 'Ft10', 'T7', 'C3', 'Cz',
                'C4', 'T8', 'Tp9', 'Cp5', 'Cp1', 'Cp2', 'Cp6', 'Tp10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    data = np.random.normal(loc=0, scale=0.00001, size=(5000, len(channels))).T
    fs = 500
else:
    import h5py
    from pynfb.postprocessing.utils import get_info
    with h5py.File(r'D:\mu_ica\mu_ica\mu_ica_S1_D3_04-21_18-16-03\experiment_data.h5') as f:
        fs, channels, p_names = get_info(f, [])
        data = f['protocol{}/raw_data'.format(p_names.index('Baseline') + 1)][:].T
    from PyQt4.QtGui import QApplication
    a = QApplication([])
    rej, spatial, top = ICADialog.get_rejection(data.T, channels, fs, mode='ica', states=None)[:3]
    data = rej.apply(data.T).T

# create info
info = mne.create_info(ch_names=channels, sfreq=fs, montage=mne.channels.read_montage(kind='standard_primed'), ch_types=['eeg' for ch in channels])

# raw instance
raw = mne.io.RawArray(data, info)
#noise_cov = mne.compute_raw_covariance(raw)
noise_cov = mne.make_ad_hoc_cov(info, verbose=None)
# forward solution
#fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, fname='fsaverage-fwd.fif', meg=False, eeg=True, mindist=5.)
fwd = mne.read_forward_solution(r'C:\Users\nsmetanin\PycharmProjects\nfb\tests\sloreta\av_brain\fsaverage\bem\fsaverage-fwd.fif', surf_ori=True)




# inverse
from mne.minimum_norm.inverse import _assemble_kernel
inv = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8, fixed=True)

# setup roi
area = True
if area:
    labels = mne.read_labels_from_annot('fsaverage', parc='aparc')
    roi_label = labels[[label.name for label in labels].index('posteriorcingulate-rh')]
    arg = None
else:
    xyz = np.array([0.9875, -0.0314, 0.1542])
    locations = src[1]['rr']
    arg = np.argmin(np.array(list(map(lambda x: np.dot(x, x), locations - xyz))))
    roi_label = None



# prepare inv
method = 'sLORETA'
inv = mne.minimum_norm.prepare_inverse_operator(inv, nave=1, lambda2=0.1, method=method)
label = None if not area else roi_label
K, noise_norm, vertno = _assemble_kernel(inv, label=roi_label, method=method, pick_ori=None)
sol = np.dot(K, data)
print(sol.shape)
if noise_norm is not None:
    sol *= noise_norm

if not area:
    sol = sol[arg//3]

plt.plot(sol.T, 'r', alpha=0.2)

plt.plot(np.mean(sol.T, axis=1))
plt.show()
#raw.set_eeg_reference()
#stc = mne.minimum_norm.apply_inverse_raw(raw, inv, 0.1, method=method, prepared=True)
#plt.plot(1e3 * stc.times, stc.data[::150, :].T)
#plt.show()

mne.extract_label_time_course()
