import pandas as pd
from pynfb.postprocessing.utils import load_data, runica2
from pynfb.signal_processing.helpers import get_outliers_mask
import numpy as np
from pynfb.inlets.montage import Montage
from mne.viz import plot_topomap
import pylab as plt

# set working dir path, which stores info.csv and extracted filters
wdir = '/home/nikolai/_Work/bci_smr_bci/'

# set experimental data path
data_dir = '/media/nikolai/D27ECFCB7ECFA697/Users/Nikolai/Desktop/bci-nfb-bci/bci_nfb_bci/bci_nfb_bci/'

##################################################################

# Set number of experiment for each experiment
n_exp = 0

##################################################################

# load experiments
experiments = pd.read_csv(wdir + 'bci-mu-bci-info.csv')

# print exp info
print(n_exp)
exp = experiments.iloc[n_exp]
desc = '{}-{}-d{}'.format(exp['subject'], {0: 'real', 1:'mock'}[exp['mock']], exp['day'])
print(exp, '\n*******************', desc, '\n*******************')

# load data
df, fs, p_names, channels = load_data('{}{}/experiment_data.h5'.format(data_dir, exp.dataset))

# drop outliers
#df = df[~get_outliers_mask(df[channels], std=4)]

# run 2 ICA analysis for Right and Left sequentially
right, left = runica2(df.loc[df['block_number'] < 13, channels], fs, channels, ['RIGHT', 'LEFT'])

# save filters
np.save(wdir + desc + '-RIGHT.npy', right)
np.save(wdir + desc + '-LEFT.npy', left)


# load filters (just for check)
right = np.load(wdir + desc + '-RIGHT.npy')
left = np.load(wdir + desc + '-LEFT.npy')

# plot filters
f, ax = plt.subplots(1, 4)
plot_topomap(right[0], Montage(channels).get_pos(), contours=0, axes=ax[0], show=False)
plot_topomap(right[1], Montage(channels).get_pos(), contours=0, axes=ax[1], show=False)
plot_topomap(left[0], Montage(channels).get_pos(), contours=0, axes=ax[2], show=False)
plot_topomap(left[1], Montage(channels).get_pos(), contours=0, axes=ax[3], show=False)
ax[0].set_title('Right spat.')
ax[1].set_title('Right topog.')
ax[2].set_title('Left spat.')
ax[3].set_title('Left topog.')
ax[0].set_ylabel(desc)

# save filters
plt.savefig(wdir + desc+'-SMR-filters.png')
plt.show()