import os
import mne

from pynfb.helpers import roi_spatial_filter as rsf

sample_data_folder = mne.datasets.sample.data_path()
subjects_dir = os.path.join(sample_data_folder, 'subjects')
Brain = mne.viz.get_brain_class()
brain = Brain('sample', hemi='both', surf='pial',
              subjects_dir=subjects_dir, size=(800, 600))
label_names_rh = ["inferiorparietal-rh", "lateraloccipital-rh"]
label_rh = rsf.get_roi_by_name(label_names_rh)
# brain.add_label(label_rh)
# brain.add_annotation('aparc.a2009s', borders=False)


import os.path as op
from mne.datasets import sample

data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
sample_dir = op.join(data_path, 'MEG', 'sample')
brain_kwargs = dict(alpha=1, background='white', cortex='low_contrast')
stc = mne.read_source_estimate(op.join(sample_dir, 'sample_audvis-meg'))
stc.crop(0.09, 0.1)
kwargs = dict(fmin=stc.data.min(), fmax=stc.data.max(), alpha=0.25,
              smoothing_steps='nearest', time=stc.times)
brain = mne.viz.Brain('sample', subjects_dir=subjects_dir, **brain_kwargs)
brain.add_label('BA4a', hemi='lh', color='red', borders=False)
brain.add_label('BA4p', hemi='lh', color='red', borders=False)
brain.add_label('BA1', hemi='lh', color='red', borders=False)
brain.add_label('BA2', hemi='lh', color='red', borders=False)
brain.add_label('BA3a', hemi='lh', color='red', borders=False)
brain.add_label('BA3b', hemi='lh', color='red', borders=False)
brain.add_label('BA3b', hemi='lh', color='red', borders=False)
brain.add_label('BA44', hemi='lh', color='purple', borders=False)

brain.add_label('BA4a', hemi='rh', color='red', borders=False)
brain.add_label('BA4p', hemi='rh', color='red', borders=False)
brain.add_label('BA1', hemi='rh', color='red', borders=False)
brain.add_label('BA2', hemi='rh', color='red', borders=False)
brain.add_label('BA3a', hemi='rh', color='red', borders=False)
brain.add_label('BA3b', hemi='rh', color='red', borders=False)
brain.add_label('BA3b', hemi='rh', color='red', borders=False)
brain.add_label('BA44', hemi='rh', color='purple', borders=False)
brain.show_view(view='dorsal')
# brain.show_view(azimuth=190, elevation=70, distance=350, focalpoint=(0, 0, 20))

# Brain = mne.viz.get_brain_class()
# mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir,
#                                         verbose=True)
# mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir,
#                                           verbose=True)
# labels = mne.read_labels_from_annot(
#     'fsaverage', 'HCPMMP1', 'lh', subjects_dir=subjects_dir)

#---------
# surfer_kwargs = dict(
#     hemi='both',
#     clim='auto', views='lateral',  # clim=dict(kind='value', lims=[8, 12, 15])
#     time_unit='s', size=(800, 800), smoothing_steps=10,
#     surface='Pial', transparent=True, alpha=0.9, colorbar=True)
# brain = stc.plot(**surfer_kwargs)
# brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
#                font_size=14)


import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()