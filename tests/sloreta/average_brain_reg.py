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
    from PyQt5.QtWidgets import QApplication
    a = QApplication([])
    rej, spatial, top = ICADialog.get_rejection(data.T, channels, fs, mode='ica', states=None)[:3]
    data = rej.apply(data.T).T

standard_montage = mne.channels.read_montage(kind='standard_1005')
standard_montage_names = [name.upper() for name in standard_montage.ch_names]
for j, channel in enumerate(channels):
    channels[j] = standard_montage.ch_names[standard_montage_names.index(channel.upper())]
# create info
info = mne.create_info(ch_names=channels, sfreq=fs, montage=mne.channels.read_montage(kind='standard_1005'), ch_types=['eeg' for ch in channels])

# raw instance
raw = mne.io.RawArray(data, info)
#raw.set_eeg_reference()
#noise_cov = mne.compute_raw_covariance(raw)
noise_cov = mne.make_ad_hoc_cov(info, verbose=None)
# forward solution
#fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, fname='fsaverage-fwd.fif', meg=False, eeg=True, mindist=5.)
fwd = mne.read_forward_solution(r'C:\Users\nsmetanin\PycharmProjects\nfb\tests\sloreta\fsaverage-fwd-1005-2.fif', surf_ori=True)




# inverse
from mne.minimum_norm.inverse import _assemble_kernel
inv = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8, fixed=True)



lambdas = [1000, 100, 10, 1, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
f, axes = plt.subplots(2, len(lambdas))
f.set_size_inches(10, 3.5)
label_name = ['caudalanteriorcingulate', 'lateraloccipital', 'posteriorcingulate'][0]
for j, lambda2 in enumerate(lambdas):
    for k, add_label in enumerate(['lh', 'rh']):
        # setup roi
        area = True
        labels = mne.read_labels_from_annot('fsaverage', parc='aparc')
        print([label.name for label in labels])
        roi_label = labels[[label.name for label in labels].index('{}-{}'.format(label_name, add_label))]
        arg = None

        # prepare inv
        method = 'sLORETA'
        inv = mne.minimum_norm.prepare_inverse_operator(inv, nave=1, lambda2=lambda2, method=method)
        label = None if not area else roi_label
        K, noise_norm, vertno = _assemble_kernel(inv, label=roi_label, method=method, pick_ori=None)
        sol = np.dot(K, data)
        print(sol.shape, noise_norm.shape)
        if noise_norm is not None:
            sol *= noise_norm


        #plt.plot(sol.T, 'r', alpha=0.2)
        #plt.plot(np.mean(sol.T, axis=1))
        #plt.show()
        #raw.set_eeg_reference()
        #stc = mne.minimum_norm.apply_inverse_raw(raw, inv, 0.1, method=method, prepared=True)
        #plt.plot(1e3 * stc.times, stc.data[::150, :].T)
        #plt.show()

        # get flip
        nvert = [len(vn) for vn in vertno]
        if label.hemi == 'both':
            # handle BiHemiLabel
            sub_labels = [label.lh, label.rh]
        else:
            sub_labels = [label]
        this_vertidx = list()
        for slabel in sub_labels:
            if roi_label.hemi == 'lh':
                this_vertno = np.intersect1d(vertno[0], roi_label.vertices)
                vertidx = np.searchsorted(vertno[0], this_vertno)
            elif roi_label.hemi == 'rh':
                this_vertno = np.intersect1d(vertno[1], roi_label.vertices)
                vertidx = nvert[0] + np.searchsorted(vertno[1], this_vertno)
            else:
                raise ValueError('label %s has invalid hemi' % label.name)
            this_vertidx.append(vertidx)
        vertidx = np.concatenate(this_vertidx)
        from mne.source_estimate import _get_label_flip
        label_flip = _get_label_flip([roi_label], [vertidx], inv['src'][:2])
        label_flip = np.array(label_flip).flatten()

        # get mean
        mean_flip = np.dot(label_flip/len(label_flip),  sol)
        #plt.plot(mean_flip, 'g')

        w = np.dot(noise_norm.flatten()*label_flip/len(label_flip), K)

        #plt.plot(np.dot(w, data), 'k--')
        #plt.figure()
        from pynfb.widgets.helpers import ch_names_to_2d_pos
        mne.viz.plot_topomap(w, info, axes=axes[k, j], show=False,contours=0)
        axes[0, j].set_title(str(lambda2))



        # back engineering flip
        #from pynfb.helpers.mne_source_estimate import extract_label_time_course
        #stc = mne.minimum_norm.apply_inverse_raw(raw, inv, 0.1, method=method, prepared=True)
        #plt.plot(mne.extract_label_time_course(stc, roi_label, inv['src'], mode='mean_flip')[0], 'k--')

axes[0, 0].set_ylabel('LH')
axes[1, 0].set_ylabel('RH')
plt.suptitle(label_name)
plt.savefig('{}.png'.format(label_name))
plt.show()

