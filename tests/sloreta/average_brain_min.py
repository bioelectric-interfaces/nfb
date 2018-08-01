import mne
import numpy as np
import pylab as plt
from pynfb.protocols.ssd.topomap_selector_ica import ICADialog
from mne.minimum_norm.inverse import _assemble_kernel

# get flip
def get_flip(label, vertno, inv):
    nvert = [len(vn) for vn in vertno]
    if label.hemi == 'both':
        sub_labels = [label.lh, label.rh]
    else:
        sub_labels = [label]
    this_vertidx = list()
    for slabel in sub_labels:
        if slabel.hemi == 'lh':
            this_vertno = np.intersect1d(vertno[0], slabel.vertices)
            vertidx = np.searchsorted(vertno[0], this_vertno)
        elif slabel.hemi == 'rh':
            this_vertno = np.intersect1d(vertno[1], slabel.vertices)
            vertidx = nvert[0] + np.searchsorted(vertno[1], this_vertno)
        else:
            raise ValueError('label %s has invalid hemi' % label.name)
        this_vertidx.append(vertidx)
    vertidx = np.concatenate(this_vertidx)
    from mne.source_estimate import _get_label_flip
    label_flip = _get_label_flip([label], [vertidx], inv['src'][:2])
    label_flip = np.array(label_flip).flatten()
    return label_flip


def get_some_data(real=False):
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
    return data, fs, channels

def get_filter(K, vertno, inv, roi_label, noise_norm):
    label_flip = get_flip(roi_label, vertno, inv)
    w = np.dot(noise_norm.flatten() * label_flip / len(label_flip), K)
    return w

# setup roi
def get_roi_by_name(name):
    labels = mne.read_labels_from_annot('fsaverage', parc='aparc')
    #print([label.name for label in labels])
    roi_label = labels[[label.name for label in labels].index(name)]
    return roi_label

def get_roi_filter(label_name, fs, channels, show=False, method='sLORETA', lambda2=1):
    info = mne.create_info(ch_names=channels, sfreq=fs, montage=mne.channels.read_montage(kind='standard_1005'), ch_types=['eeg' for ch in channels])
    mne.utils.set_config("SUBJECTS_DIR", 'av_brain', set_env=True)
    noise_cov = mne.make_ad_hoc_cov(info, verbose=None)
    fwd = mne.read_forward_solution(r'C:\Users\nsmetanin\PycharmProjects\nfb\tests\sloreta\fsaverage-fwd-1005-1.fif', surf_ori=True)
    inv = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8, fixed=True)
    inv = mne.minimum_norm.prepare_inverse_operator(inv, nave=1, lambda2=lambda2, method=method)
    roi_label = get_roi_by_name(label_name)
    K, noise_norm, vertno = _assemble_kernel(inv, label=roi_label, method=method, pick_ori=None)
    w = get_filter(K, vertno, inv, roi_label, noise_norm)
    if show:
        mne.viz.plot_topomap(w, info)
    return w


if __name__ == '__main__':
    # get data
    data, fs, channels = get_some_data(real=False)
    label_name = 'posteriorcingulate-rh'
    w = get_roi_filter(label_name, fs, channels, show=True)
    plt.figure()
    plt.plot(np.dot(w, data), 'k')
    plt.show()
